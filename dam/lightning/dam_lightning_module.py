from typing import Any, Dict, List, Tuple
from typing import Any, Dict, List, Tuple
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
import numpy as np
import inspect

from .. import src as dam


class DamLightning(LightningModule):
    def __init__(
        self,
        *,
        lr: float,
        image_loss: str,
        lambda_img: float,
        kappa: float,
        beta: float,
        kl_weight_prior: float,
        kl_weight_post: float,  # set 0 for original setup if desired
        bidir: bool,
        latent_dim: int,
        enc_nf: List[int],
        dec_nf: List[int],
        int_steps: int,
        int_downsize: int,
        conv_per_level: int,
        num_organs: int = 1,
        cudnn_nondet: bool = False,
        evaluate_every_n_epochs: int = 5,
        evaluate_num_pairs: int = 8,
        vis_slice_mode: str = "mid",          # {"mid","index","mip"}
        vis_slice_index: int | None = None,   # used if vis_slice_mode == "index"
        vis_channel: int = 0,
        vis_normalize_per_image: bool = True,
    ):
        super().__init__()

        sig = inspect.signature(self.__init__)
        allowed = {name for name, p in sig.parameters.items() if name != "self"}
        hp = {k: v for k, v in locals().items()
              if k in allowed and not k.startswith("_")}
        self.save_hyperparameters(hp)

        # Core settings
        self.model = None
        self._image_loss_name = image_loss
        self.lr = lr
        self.bidir = bidir
        self.num_organs = num_organs

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = not cudnn_nondet

        # Eval / visualization controls
        self.evaluate_every_n_epochs = max(1, int(evaluate_every_n_epochs))
        self.evaluate_num_pairs = int(evaluate_num_pairs)
        self.vis_slice_mode = vis_slice_mode
        self.vis_slice_index = vis_slice_index
        self.vis_channel = int(vis_channel)
        self.vis_normalize_per_image = bool(vis_normalize_per_image)

        # How many example pairs we’ve logged in the current eval epoch
        self.register_buffer("_eval_pairs_logged", torch.tensor(0.0))

    # -------------------------- Setup & losses --------------------------
    def build_model(self, inshape):
        self.model = dam.networks.DamBase(
            inshape=inshape,
            latent_dim=self.hparams.latent_dim,
            nb_unet_features=[self.hparams.enc_nf, self.hparams.dec_nf],
            int_steps=self.hparams.int_steps,
            int_downsize=self.hparams.int_downsize,
            nb_unet_conv_per_level=self.hparams.conv_per_level,
            bidir=self.hparams.bidir,
        )

    def setup(self, stage: Any = None):
        if self.model is None:
            self.build_model(self.trainer.datamodule.inshape)

        if self._image_loss_name == "ncc":
            self.img_loss = dam.losses.NCC(device=self.device).loss
        elif self._image_loss_name == "mse":
            self.img_loss = dam.losses.MSE().loss
        else:
            raise ValueError(f"Invalid image_loss: {self._image_loss_name}")

        self.dice_loss = dam.losses.Dice().loss
        self.grad_loss = dam.losses.Grad("l2", loss_mult=self.hparams.int_downsize).loss
        self.prior_KL = dam.losses.KL(prior=True).loss
        self.post_KL = dam.losses.KL(prior=False).loss

    # -------------------------- Small utilities --------------------------

    def _wandb(self) -> WandbLogger | None:
        """Return the first WandbLogger, if present."""
        for lg in self.trainer.loggers or []:
            if isinstance(lg, WandbLogger):
                return lg
        return None

    @staticmethod
    def _linear_warmup(epoch: int, target: float, start: int, end: int) -> float:
        if epoch <= start:
            return 0.0
        if epoch >= end:
            return float(target)
        return float(target) * (epoch - start) / max(1, end - start)

    @staticmethod
    def _to01_np(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)
    
    @staticmethod
    def _to01_t(x: torch.Tensor) -> torch.Tensor:
        mn = x.amin()
        mx = x.amax()
        return (x - mn) / (mx - mn + 1e-8)

    @staticmethod
    def _psnr01(gt01: np.ndarray, rc01: np.ndarray) -> float:
        mse = float(np.mean((gt01 - rc01) ** 2))
        return float(10.0 * np.log10(1.0 / (mse + 1e-12)))

    @staticmethod
    def _ncc(gt01: np.ndarray, rc01: np.ndarray) -> float:
        g = gt01 - gt01.mean()
        r = rc01 - rc01.mean()
        denom = np.sqrt((g ** 2).sum() * (r ** 2).sum()) + 1e-12
        return float((g * r).sum() / denom)
    
    @staticmethod
    def _ncc_t(gt: torch.Tensor, rc: torch.Tensor) -> float:
        """NCC over full 3D tensor (any shape), returns float."""
        g = gt - gt.mean()
        r = rc - rc.mean()
        denom = (g.square().sum().sqrt() * r.square().sum().sqrt()) + 1e-12
        return float((g * r).sum() / denom)

    def _pick_slice(self, vol: torch.Tensor) -> torch.Tensor:

        c = self.vis_channel
        volc = vol[:, c]  # [B, X, Y, Z]

        if self.vis_slice_mode == "mip":
            # Max-intensity projection along Z (last dim)
            img = volc.max(dim=-3).values  # [B, X, Y]
        else:
            Z = volc.shape[-3]
            if self.vis_slice_mode == "mid" or self.vis_slice_index is None:
                z = Z // 2
            else:
                z = max(0, min(Z - 1, int(self.vis_slice_index)))
            img = volc[..., z, :, :]
        return img

    def _normalize01(self, img: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalization over [X, Y]."""
        if not self.vis_normalize_per_image:
            return img
        bmin = img.amin(dim=(1, 2), keepdim=True)
        bmax = img.amax(dim=(1, 2), keepdim=True)
        return (img - bmin) / (bmax - bmin + 1e-8)

    def _prepare_batch(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        inputs, y_true, pmasks, rmasks, pamask, ramask = batch
        return dict(
            inputs=inputs.float().permute(0, 4, 1, 2, 3),
            y_true=y_true.float().permute(0, 4, 1, 2, 3),
            pmasks=pmasks.to(torch.int64).permute(0, 4, 1, 2, 3),
            rmasks=rmasks.to(torch.int64).permute(0, 4, 1, 2, 3),
            pamask=pamask.to(torch.int64).permute(0, 4, 1, 2, 3),
            ramask=ramask.to(torch.int64).permute(0, 4, 1, 2, 3),
        )

    def _model_outputs_to_dict(self, y_pred, bidir: bool) -> Dict[str, Any]:
        """Map DamBase tuple outputs to named entries."""
        if bidir:
            rep, inv, mrep, minv, svf, param = y_pred
            return {"rep": rep, "inv": inv, "mrep": mrep, "minv": minv, "svf": svf, "param": param}
        else:
            rep, mrep, svf, param = y_pred
            return {"rep": rep, "mrep": mrep, "svf": svf, "param": param}

    # -------------------------- Visuals & metric panels --------------------------
    def _build_pair_panel(
        self, 
        planning2d: torch.Tensor, 
        gt2d: torch.Tensor, 
        rc2d: torch.Tensor,
        mask2d: torch.Tensor = None
    ) -> tuple[wandb.Image, Dict[str, float]]:
        """
        Build a panel figure with:
        - planning, GT, recon, |diff|
        - if mask provided: adds GT+mask overlay panel
        Inputs are 2D tensors on CPU (normalized to [0,1] recommended).
        """
        pl = planning2d.numpy()
        gt = gt2d.numpy()
        rc = rc2d.numpy()

        gt01 = self._to01_np(gt)
        rc01 = self._to01_np(rc)
        diff = np.abs(gt01 - rc01)

        mse = float(np.mean((gt01 - rc01) ** 2))
        psnr = self._psnr01(gt01, rc01)
        ncc = self._ncc(gt01, rc01)

        # Create figure - add extra panel if mask provided
        ncols = 5 if mask2d is not None else 4
        fig, axs = plt.subplots(1, ncols, figsize=(3*ncols, 3), constrained_layout=True)
        
        # Standard panels
        axs[0].imshow(pl, cmap="gray"); axs[0].set_title("Planning", fontsize=10); axs[0].axis("off")
        axs[1].imshow(gt, cmap="gray"); axs[1].set_title("GT Repeat", fontsize=10); axs[1].axis("off")
        axs[2].imshow(rc, cmap="gray"); axs[2].set_title("Recon Repeat", fontsize=10); axs[2].axis("off")
        im = axs[3].imshow(diff, cmap="magma")
        axs[3].set_title(f"|GT−Recon|\nMSE={mse:.4f}  PSNR={psnr:.2f}  NCC={ncc:.3f}", fontsize=9)
        axs[3].axis("off")
        plt.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)

        # Add mask overlay if provided
        if mask2d is not None:
            mask_np = mask2d.numpy()
            # Create red-tinted overlay where mask is active
            mask_overlay = np.stack([
                gt01,  # R channel: use GT
                gt01 * (1 - mask_np),  # G channel: GT dimmed by mask
                gt01 * (1 - mask_np),  # B channel: GT dimmed by mask
            ], axis=-1)
            axs[4].imshow(mask_overlay)
            axs[4].set_title("GT + Anatomy Mask", fontsize=10)
            axs[4].axis("off")

        panel = wandb.Image(fig)
        plt.close(fig)
        return panel, {"mse": mse, "psnr": psnr, "ncc": ncc}

    def _maybe_log_examples(
        self,
        *,
        stage: str,
        inputs: torch.Tensor,
        y_true: torch.Tensor,
        pred: Dict[str, torch.Tensor],
        limit: int,
        eval_gate: bool,
        anatomy_mask: torch.Tensor = None,  # [B,C,X,Y,Z] anatomy mask
    ) -> None:
        """Added anatomy_mask parameter to visualize mask overlay"""
        if not eval_gate:
            return
        wb = self._wandb()
        if wb is None:
            return

        k = min(limit, y_true.size(0))
        if k <= 0:
            return

        recon = pred["rep"][:k]
        yt_k = y_true[:k]
        in_k = inputs[:k]
        mask_k = anatomy_mask[:k] if anatomy_mask is not None else None

        panels = []
        for i in range(k):
            pl2d = self._normalize01(self._pick_slice(in_k[i:i+1]))[0].detach().cpu()
            gt2d = self._normalize01(self._pick_slice(yt_k[i:i+1]))[0].detach().cpu()
            rc2d = self._normalize01(self._pick_slice(recon[i:i+1]))[0].detach().cpu()
            # Get corresponding mask slice if available
            mask2d = self._pick_slice(mask_k[i:i+1])[0].detach().cpu() if mask_k is not None else None
            panel, _ = self._build_pair_panel(pl2d, gt2d, rc2d, mask2d)
            panels.append(panel)

        wb.experiment.log({f"{stage}/examples_gt_vs_recon": panels, "epoch": self.current_epoch})

    # -------------------------- Core step --------------------------

    def _step(self, batch: Tuple[torch.Tensor, ...], stage: str) -> Dict[str, torch.Tensor]:
        """
        batch: (planning, repeat, planning_mask, repeat_mask, planning_anat_mask, repeat_anat_mask) as [B,X,Y,Z,C]
        """
        b = self._prepare_batch(batch)
        inputs, y_true = b["inputs"], b["y_true"]
        pmasks, rmasks = b["pmasks"], b["rmasks"]
        pamask = b.get("pamask", None)
        ramask = b.get("ramask", None)

        # forward pass
        y_pred = self.model(inputs, y_true, pmasks, rmasks)
        pred = self._model_outputs_to_dict(y_pred, self.bidir)

        # one-hot masks for Dice
        pm_bin = self.model.to_binary(pmasks, num_organs=self.num_organs)
        rm_bin = self.model.to_binary(rmasks, num_organs=self.num_organs)

        # compose losses
        comps: Dict[str, torch.Tensor] = {}
        if self.bidir:
            comps["img_fwd"]  = self.img_loss(y_true,  pred["rep"], mask=ramask) * self.hparams.lambda_img
            comps["img_bwd"]  = self.img_loss(inputs, pred["inv"], mask=pamask) * self.hparams.lambda_img
            comps["dice_fwd"] = self.dice_loss(rm_bin, pred["mrep"]) * self.hparams.kappa
            comps["dice_bwd"] = self.dice_loss(pm_bin, pred["minv"]) * self.hparams.kappa
        else:
            comps["img"]  = self.img_loss(y_true,  pred["rep"], mask=ramask)   * self.hparams.lambda_img
            comps["dice"] = self.dice_loss(rm_bin, pred["mrep"]) * self.hparams.kappa

        comps["grad"] = self.grad_loss(None, pred["svf"]) * self.hparams.beta

        # Warmups
        kp = self._linear_warmup(self.current_epoch, float(self.hparams.kl_weight_prior), 0, 20)
        ko = self._linear_warmup(self.current_epoch, float(self.hparams.kl_weight_post),  5, 25)
        comps["kl_prior"] = self.prior_KL(None, pred["param"]) * kp
        comps["kl_post"]  = self.post_KL(None,  pred["param"]) * ko

        total = torch.stack(list(comps.values())).sum()

        # Log per-component + total
        for name, val in comps.items():
            self.log(f"{stage}/loss/{name}", val, on_step=False, on_epoch=True, sync_dist=True)

        self.log(f"{stage}/loss_total", total, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": total, "inputs": inputs, "y_true": y_true, "pred": pred, "masks": ramask}

    # -------------------------- Lightning hooks --------------------------

    def training_step(self, batch, batch_idx):
        # main step
        out = self._step(batch, stage="train")

        # gated example logging on train
        do_eval_epoch = ((self.current_epoch + 1) % self.evaluate_every_n_epochs == 0)
        remain = int(self.evaluate_num_pairs - self._eval_pairs_logged.item())
        if do_eval_epoch and remain > 0:
            self._maybe_log_examples(
                stage="train",
                inputs=out["inputs"],
                y_true=out["y_true"],
                pred=out["pred"],
                limit=remain,
                eval_gate=True,
                anatomy_mask=out["masks"],
            )
            # advance pair counter
            k_used = min(remain, out["y_true"].size(0))
            self._eval_pairs_logged += torch.as_tensor(float(k_used), device=self._eval_pairs_logged.device)

        return out["loss"]

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.evaluate_every_n_epochs == 0:
            self._eval_pairs_logged.zero_()

    def validation_step(self, batch, batch_idx):
        out = self._step(batch, stage="val")

        # gated example logging on val (independent of train gating)
        do_eval_epoch = ((self.current_epoch + 1) % self.evaluate_every_n_epochs == 0)
        self._maybe_log_examples(
            stage="val",
            inputs=out["inputs"],
            y_true=out["y_true"],
            pred=out["pred"],
            limit=self.evaluate_num_pairs,
            eval_gate=do_eval_epoch,
            anatomy_mask=out["masks"],
        )

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.evaluate_every_n_epochs == 0:
            self._eval_pairs_logged.zero_()

    # ------------------------------ Optimizer --------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ------------------------------- Inference --------------------------------

    def forward(self, inputs, y_true, pmasks, rmasks):
        return self.model(inputs, y_true, pmasks, rmasks)
