from typing import Any, Dict, List, Tuple
from typing import Any, Dict, List, Tuple
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
import numpy as np

from .. import src as dam
from ..src.metrics import grad_mae_3d
from ..utils.elastix import elastix_register_and_resample


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
        self.save_hyperparameters()

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

        # Stage-specific accumulators (device-safe)
        self.register_buffer("_train_gradmae_sum", torch.tensor(0.0))
        self.register_buffer("_train_gradmae_count", torch.tensor(0.0))
        self.register_buffer("_val_gradmae_sum", torch.tensor(0.0))
        self.register_buffer("_val_gradmae_count", torch.tensor(0.0))
        self.register_buffer("_train_vs_elx_ncc_sum",   torch.tensor(0.0))
        self.register_buffer("_train_vs_elx_ncc_count", torch.tensor(0.0))
        self.register_buffer("_val_vs_elx_ncc_sum",     torch.tensor(0.0))
        self.register_buffer("_val_vs_elx_ncc_count",   torch.tensor(0.0))

        # How many example pairs we’ve logged in the current eval epoch
        self.register_buffer("_eval_pairs_logged", torch.tensor(0.0))

    # -------------------------- Setup & losses --------------------------

    def setup(self, stage: Any = None):
        self.hparams.inshape = self.trainer.datamodule.inshape  # (X, Y, Z)

        if self.model is None:
            self.model = dam.networks.DamBase(
                inshape=self.hparams.inshape,
                latent_dim=self.hparams.latent_dim,
                nb_unet_features=[self.hparams.enc_nf, self.hparams.dec_nf],
                int_steps=self.hparams.int_steps,
                int_downsize=self.hparams.int_downsize,
                nb_unet_conv_per_level=self.hparams.conv_per_level,
                bidir=self.hparams.bidir,
            )

        if self._image_loss_name == "ncc":
            self.img_loss = dam.losses.NCC(device=self.device, use_weights=True).loss
        elif self._image_loss_name == "mse":
            self.img_loss = dam.losses.MSE(use_weights=True).loss
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
        inputs, y_true, pmasks, rmasks = batch
        return dict(
            inputs=inputs.float().permute(0, 4, 1, 2, 3),
            y_true=y_true.float().permute(0, 4, 1, 2, 3),
            pmasks=pmasks.to(torch.int64).permute(0, 4, 1, 2, 3),
            rmasks=rmasks.to(torch.int64).permute(0, 4, 1, 2, 3),
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
    def _eval_elastix_vs_model(
        self, planning_3d: torch.Tensor, gt_3d: torch.Tensor
    ) -> dict:
        """
        planning_3d, gt_3d: [X, Y, Z] torch tensors on CPU.
        Returns dict with Elastix metrics
        """
        pl_np = planning_3d.numpy()
        gt_np = gt_3d.numpy()

        # 1) Elastix recon (planning -> fixed=GT)
        elx_rc_np = elastix_register_and_resample(pl_np, gt_np)  # [X,Y,Z] np.float64
        elx_rc = torch.from_numpy(elx_rc_np)

        # 2) Compute image metrics
        def to01_t(x: torch.Tensor) -> torch.Tensor:
            mn, mx = x.amin(), x.amax()
            return (x - mn) / (mx - mn + 1e-8)

        gt01  = to01_t(gt_3d)
        elx01 = to01_t(elx_rc)

        g  = (gt01 - gt01.mean())
        e  = (elx01 - elx01.mean())
        ncc_elx = float((g*e).sum() / (g.square().sum().sqrt() * e.square().sum().sqrt() + 1e-12))

        return dict(ncc_elx=ncc_elx, elx_rc=elx_rc)

    def _build_pair_panel(self, planning2d: torch.Tensor, gt2d: torch.Tensor, rc2d: torch.Tensor) -> tuple[wandb.Image, Dict[str, float]]:
        """
        Build a 5-panel figure (planning, GT, recon, overlay, |diff|) and return
        a WandB image plus per-pair metrics (MSE, PSNR, NCC).
        Inputs are 2D tensors on CPU (normalized to [0,1] recommended).
        """
        pl = planning2d.numpy()
        gt = gt2d.numpy()
        rc = rc2d.numpy()

        gt01 = self._to01_np(gt)
        rc01 = self._to01_np(rc)
        overlay = np.stack([rc01, gt01, rc01], axis=-1)
        diff = np.abs(gt01 - rc01)

        mse = float(np.mean((gt01 - rc01) ** 2))
        psnr = self._psnr01(gt01, rc01)
        ncc = self._ncc(gt01, rc01)

        fig, axs = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
        axs[0].imshow(pl, cmap="gray"); axs[0].set_title("Planning", fontsize=10); axs[0].axis("off")
        axs[1].imshow(gt, cmap="gray"); axs[1].set_title("GT Repeat", fontsize=10); axs[1].axis("off")
        axs[2].imshow(rc, cmap="gray"); axs[2].set_title("Recon Repeat", fontsize=10); axs[2].axis("off")
        axs[3].imshow(overlay); axs[3].set_title("Overlay (G=GT, M=Recon)", fontsize=10); axs[3].axis("off")
        im = axs[4].imshow(diff, cmap="magma")
        axs[4].set_title(f"|GT−Recon|\nMSE={mse:.4f}  PSNR={psnr:.2f}  NCC={ncc:.3f}", fontsize=9)
        axs[4].axis("off")
        plt.colorbar(im, ax=axs[4], fraction=0.046, pad=0.04)

        panel = wandb.Image(fig)
        plt.close(fig)
        return panel, {"mse": mse, "psnr": psnr, "ncc": ncc}

    def _maybe_log_examples(
        self,
        *,
        stage: str,                               # "train" or "val"
        inputs: torch.Tensor,                     # [B, C, X, Y, Z]
        y_true: torch.Tensor,                     # [B, C, X, Y, Z]
        pred: Dict[str, torch.Tensor],            # outputs dict
        limit: int,
        eval_gate: bool,                          # whether this epoch is an eval-logging epoch
        accum_key_prefix: str,                    # e.g., "train" or "val" for accumulators
    ) -> None:
        """
        Optionally log example panels + grad-MAE, and accumulate stage metrics.
        Uses only up to 'limit' samples from the current batch.
        """
        if not eval_gate:
            return
        wb = self._wandb()
        if wb is None:
            return

        # Select k samples
        k = min(limit, y_true.size(0))
        if k <= 0:
            return

        recon = pred["rep"][:k]                  # [k,1,X,Y,Z]
        yt_k  = y_true[:k]                       # [k,1,X,Y,Z]
        in_k  = inputs[:k]                       # [k,C,X,Y,Z]

        # Compute and accumulate grad-MAE on these k samples (mean across those k)
        gmae = grad_mae_3d(yt_k, recon)          # scalar
        k_tensor = torch.tensor(float(k), device=gmae.device)
        if accum_key_prefix == "train":
            self._train_gradmae_sum   += gmae * k_tensor
            self._train_gradmae_count += k_tensor
        else:
            self._val_gradmae_sum   += gmae * k_tensor
            self._val_gradmae_count += k_tensor

        # Build panels
        panels = []
        for i in range(k):
            
            # Elastix registration. Commented out for speed; enable if desired.
            """
            c = self.vis_channel
            planning3d = in_k[i, c].detach().cpu()
            gt3d       = yt_k[i, 0].detach().cpu()
            model3d    = recon[i, 0].detach().cpu()

            elx_np = elastix_register_and_resample(planning3d.numpy(), gt3d.numpy())
            elx3d  = torch.from_numpy(elx_np)

            gt01    = self._to01_t(gt3d)
            m01     = self._to01_t(model3d)
            e01     = self._to01_t(elx3d)

            ncc_m   = self._ncc_t(gt01, m01)
            ncc_e   = self._ncc_t(gt01, e01)
            delta_ncc = ncc_m - ncc_e  
            

            if accum_key_prefix == "train":
                self._train_vs_elx_ncc_sum   += torch.tensor(delta_ncc, device=self._train_vs_elx_ncc_sum.device)
                self._train_vs_elx_ncc_count += torch.tensor(1.0,       device=self._train_vs_elx_ncc_count.device)
            else:
                self._val_vs_elx_ncc_sum   += torch.tensor(delta_ncc, device=self._val_vs_elx_ncc_sum.device)
                self._val_vs_elx_ncc_count += torch.tensor(1.0,       device=self._val_vs_elx_ncc_count.device)
            """
            
            pl2d = self._normalize01(self._pick_slice(in_k[i:i+1]))[0].detach().cpu()
            gt2d = self._normalize01(self._pick_slice(yt_k[i:i+1]))[0].detach().cpu()
            rc2d = self._normalize01(self._pick_slice(recon[i:i+1]))[0].detach().cpu()
            panel, _ = self._build_pair_panel(pl2d, gt2d, rc2d)
            panels.append(panel)

        # Log with stage-specific key + epoch
        wb.experiment.log({f"{stage}/examples_gt_vs_recon": panels, "epoch": self.current_epoch})

    # -------------------------- Core step --------------------------

    def _step(self, batch: Tuple[torch.Tensor, ...], stage: str) -> Dict[str, torch.Tensor]:
        """
        batch: (planning, repeat, planning_mask, repeat_mask) as [B, X, Y, Z, C]
        """
        b = self._prepare_batch(batch)
        inputs, y_true, pmasks, rmasks = b["inputs"], b["y_true"], b["pmasks"], b["rmasks"]

        # forward pass
        y_pred = self.model(inputs, y_true, pmasks, rmasks)
        pred = self._model_outputs_to_dict(y_pred, self.bidir)

        # one-hot masks for Dice
        pm_bin = self.model.to_binary(pmasks, num_organs=self.num_organs)
        rm_bin = self.model.to_binary(rmasks, num_organs=self.num_organs)

        # compose losses
        comps: Dict[str, torch.Tensor] = {}
        if self.bidir:
            comps["img_fwd"]  = self.img_loss(y_true,  pred["rep"]) * self.hparams.lambda_img
            comps["img_bwd"]  = self.img_loss(inputs, pred["inv"]) * self.hparams.lambda_img
            comps["dice_fwd"] = self.dice_loss(rm_bin, pred["mrep"]) * self.hparams.kappa
            comps["dice_bwd"] = self.dice_loss(pm_bin, pred["minv"]) * self.hparams.kappa
        else:
            comps["img"]  = self.img_loss(y_true,  pred["rep"])   * self.hparams.lambda_img
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

        return {"loss": total, "inputs": inputs, "y_true": y_true, "pred": pred}

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
                accum_key_prefix="train",
            )
            # advance pair counter
            k_used = min(remain, out["y_true"].size(0))
            self._eval_pairs_logged += torch.as_tensor(float(k_used), device=self._eval_pairs_logged.device)

        return out["loss"]

    def on_train_epoch_end(self):
        # At the end of an eval epoch for train: log aggregated grad-MAE, then reset
        if (self.current_epoch + 1) % self.evaluate_every_n_epochs == 0:
            if self._train_gradmae_count.item() > 0:
                gmae_mean = self._train_gradmae_sum / self._train_gradmae_count.clamp_min(1.0)
                self.log("train/grad_mae3d", gmae_mean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if self._train_vs_elx_ncc_count.item() > 0:
                mean_delta_ncc = self._train_vs_elx_ncc_sum / self._train_vs_elx_ncc_count.clamp_min(1.0)
                self.log("train/vs_elastix/delta_ncc_mean", mean_delta_ncc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            # reset for next eval window
            self._train_vs_elx_ncc_sum.zero_()
            self._train_vs_elx_ncc_count.zero_()
            self._train_gradmae_sum.zero_()
            self._train_gradmae_count.zero_()
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
            accum_key_prefix="val",
        )

    def on_validation_epoch_end(self):
        # Log aggregated grad-MAE for val if we recorded any, then reset
        if (self.current_epoch + 1) % self.evaluate_every_n_epochs == 0:
            if self._val_vs_elx_ncc_count.item() > 0:
                mean_delta_ncc = self._val_vs_elx_ncc_sum / self._val_vs_elx_ncc_count.clamp_min(1.0)
                self.log("val/vs_elastix/delta_ncc_mean", mean_delta_ncc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if self._val_gradmae_count.item() > 0:
                gmae_mean = self._val_gradmae_sum / self._val_gradmae_count.clamp_min(1.0)
                self.log("val/grad_mae3d", gmae_mean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_gradmae_sum.zero_()
            self._val_gradmae_count.zero_()
            self._val_vs_elx_ncc_sum.zero_()
            self._val_vs_elx_ncc_count.zero_()

    # ------------------------------ Optimizer --------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ------------------------------- Inference --------------------------------

    def forward(self, inputs, y_true, pmasks, rmasks):
        return self.model(inputs, y_true, pmasks, rmasks)
