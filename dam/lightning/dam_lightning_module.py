from typing import Any, Dict, List, Tuple, Optional
import torch
from lightning.pytorch import LightningModule
import wandb
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np

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
        kl_weight_post: float, # Just set to 0 for original setup
        bidir: bool,
        latent_dim: int,
        enc_nf: List[int],
        dec_nf: List[int],
        int_steps: int,
        int_downsize: int,
        conv_per_level: int,
        # prior: bool,
        num_organs: int = 1,
        cudnn_nondet: bool = False,
        log_images_every_n_epochs: int = 5,
        vis_num_pairs: int = 8,
        vis_slice_mode: str = "mid",
        vis_slice_index: int | None = None,
        vis_channel: int = 0,
        vis_normalize_per_image: bool = True,
    ):
        super().__init__()
        # Save all init args to checkpoint
        self.save_hyperparameters()

        self.model = None
        self._image_loss_name = image_loss

        self.lr = lr
        self.bidir = bidir
        self.num_organs = num_organs

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = not cudnn_nondet

        # store vis cfg
        self.log_images_every_n_epochs = max(1, int(log_images_every_n_epochs))
        self.vis_num_pairs = int(vis_num_pairs)
        self.vis_slice_mode = vis_slice_mode
        self.vis_slice_index = vis_slice_index
        self.vis_channel = int(vis_channel)
        self.vis_normalize_per_image = bool(vis_normalize_per_image)
        # cache for one epoch
        self._val_vis_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _model_outputs_to_dict(self, y_pred, bidir: bool) -> Dict[str, Any]:
        """
        Map your tuple outputs to named entries; no change to DamBase needed.
        """
        if bidir:
            rep, inv, mrep, minv, svf, param = y_pred
            return {"rep": rep, "inv": inv, "mrep": mrep, "minv": minv, "svf": svf, "param": param}
        else:
            rep, mrep, svf, param = y_pred
            return {"rep": rep, "mrep": mrep, "svf": svf, "param": param}
        
    def _pick_slice(self, vol: torch.Tensor) -> torch.Tensor:
        """
        vol: [B, C, X, Y, Z]
        returns: [B, X, Y] selected slice or MIP
        """
        c = self.vis_channel
        volc = vol[:, c]  # [B, X, Y, Z]

        if self.vis_slice_mode == "mip":
            # max-intensity projection along Z
            img = volc.max(dim=-3).values  # [B, X, Y]
        else:
            # mid or index
            Z = volc.shape[-3]
            z = Z // 2 if (self.vis_slice_mode == "mid" or self.vis_slice_index is None) else int(self.vis_slice_index)
            z = max(0, min(Z - 1, z))
            img = volc[:, z]  # [B, X, Y]
        return img
    
    def _normalize01(self, img: torch.Tensor) -> torch.Tensor:
        if not self.vis_normalize_per_image:
            return img
        # per-image min-max
        bmin = img.amin(dim=(1, 2), keepdim=True)
        bmax = img.amax(dim=(1, 2), keepdim=True)
        return (img - bmin) / (bmax - bmin + 1e-8)
    
    def _collect_pairs(self, inputs, y_true, y_pred):
        """
        inputs/y_true: [B, C, X, Y, Z]
        y_pred: either list/tuple or tensor; we use the reconstructed repeat
        Collect up to remaining pairs for this epoch.
        """
        # pick the recon tensor
        recon = y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred  # <-- pick recon tensor
        
        plan2d = self._normalize01(self._pick_slice(inputs)).detach().cpu()
        gt2d = self._normalize01(self._pick_slice(y_true)).detach().cpu()
        rc2d = self._normalize01(self._pick_slice(recon)).detach().cpu()

        # how many more do we need? 
        take = max(0, min(self.vis_num_pairs, gt2d.shape[0]))
        for i in range(take):
            # each item is (GT, Recon) as [X, Y] tensors
            self._val_vis_pairs.append((plan2d[i], gt2d[i], rc2d[i]))

    def _log_pairs_to_wandb(self):
        if not self.trainer.is_global_zero or len(self._val_vis_pairs) == 0:
            self._val_vis_pairs.clear()
            return
        wb: WandbLogger | None = None
        for lg in self.trainer.loggers or []:
            if isinstance(lg, WandbLogger):
                wb = lg
                break
        if wb is None:
            self._val_vis_pairs.clear()
            return
        
        def _to01(x: torch.Tensor) -> np.ndarray:
            """min-max to [0,1] for display."""
            x = x.numpy()
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)

        def _psnr(gt01: np.ndarray, rc01: np.ndarray) -> float:
            mse = np.mean((gt01 - rc01) ** 2)
            return float(10.0 * np.log10(1.0 / (mse + 1e-12)))

        def _ncc(gt01: np.ndarray, rc01: np.ndarray) -> float:
            g = gt01 - gt01.mean()
            r = rc01 - rc01.mean()
            denom = np.sqrt((g**2).sum() * (r**2).sum()) + 1e-12
            return float((g * r).sum() / denom)

        panels = []
        for idx, (pl, gt, rc) in enumerate(self._val_vis_pairs):
            # Normalize each image to [0,1] for fair visual comparison
            gt01 = _to01(gt)
            rc01 = _to01(rc)

            # Build overlay: GT->G channel, Recon->R+B (magenta)
            overlay = np.stack([rc01, gt01, rc01], axis=-1)  # H×W×3

            # Diff heatmap (absolute error)
            diff = np.abs(gt01 - rc01)

            # Metrics
            mse  = float(np.mean((gt01 - rc01)**2))
            psnr = _psnr(gt01, rc01)
            ncc  = _ncc(gt01, rc01)

            fig, axs = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
            axs[0].imshow(pl.numpy(), cmap="gray"); axs[0].set_title("Planning", fontsize=10); axs[0].axis("off")
            axs[1].imshow(gt.numpy(), cmap="gray"); axs[1].set_title("GT Repeat", fontsize=10); axs[1].axis("off")
            axs[2].imshow(rc.numpy(), cmap="gray"); axs[2].set_title("Recon Repeat", fontsize=10); axs[2].axis("off")

            axs[3].imshow(overlay); axs[3].set_title("Overlay (G=GT, M=Recon)", fontsize=10); axs[3].axis("off")

            im = axs[4].imshow(diff, cmap="magma")
            axs[4].set_title(f"Diff |GT−Recon|\nMSE={mse:.4f}  PSNR={psnr:.2f}  NCC={ncc:.3f}", fontsize=9)
            axs[4].axis("off")
            # optional tiny colorbar
            plt.colorbar(im, ax=axs[4], fraction=0.046, pad=0.04)

            panels.append(wandb.Image(fig, caption=f"pair {idx}"))
            plt.close(fig)

        wb.experiment.log({ "train/examples_gt_vs_recon": panels, "epoch": self.current_epoch })
        self._val_vis_pairs.clear()
    # -------------------------- Lightning lifecycle --------------------------

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

    # ------------------------------- Train / Val ------------------------------
    def _step(self, batch: Tuple[torch.Tensor, ...], stage: str) -> Dict[str, torch.Tensor]:
        """
        batch: (planning, repeat, planning_mask, repeat_mask)
               arrays come from your Dataset as [B, X, Y, Z, C]; we permute to [B, C, X, Y, Z]
        """
        inputs, y_true, pmasks, rmasks = batch

        # format tensors
        inputs = inputs.float().permute(0, 4, 1, 2, 3)
        y_true = y_true.float().permute(0, 4, 1, 2, 3)
        pmasks = pmasks.to(torch.int64).permute(0, 4, 1, 2, 3)
        rmasks = rmasks.to(torch.int64).permute(0, 4, 1, 2, 3)

        # forward pass
        y_pred = self.model(inputs, y_true, pmasks, rmasks)
        pred = self._model_outputs_to_dict(y_pred, self.bidir)

        # one-hot masks
        pmasks = self.model.to_binary(pmasks, num_organs=self.num_organs)
        rmasks = self.model.to_binary(rmasks, num_organs=self.num_organs)

        comps: Dict[str, torch.Tensor] = {}

        if self.bidir:
            # image (planning->repeat) and (repeat->planning)
            comps["img_fwd"]  = self.img_loss(y_true,  pred["rep"]) * self.hparams.lambda_img
            comps["img_bwd"]  = self.img_loss(inputs, pred["inv"]) * self.hparams.lambda_img
            # dice for warped masks
            comps["dice_fwd"] = self.dice_loss(rmasks, pred["mrep"]) * self.hparams.kappa
            comps["dice_bwd"] = self.dice_loss(pmasks, pred["minv"]) * self.hparams.kappa
        else:
            comps["img"]  = self.img_loss(y_true,  pred["rep"])   * self.hparams.lambda_img
            comps["dice"] = self.dice_loss(rmasks, pred["mrep"]) * self.hparams.kappa

        comps["grad"] = self.grad_loss(None, pred["svf"]) * self.hparams.beta

        def linear_warmup(epoch, target, start, end):
            if epoch <= start: return 0.0
            if epoch >= end:   return target
            return target * (epoch - start) / max(1, end - start)

        comps["kl_prior"] = self.prior_KL(None, pred["param"]) * linear_warmup(self.current_epoch, self.hparams.kl_weight_prior, 0, 20)
        comps["kl_post"] = self.post_KL(None, pred["param"]) * linear_warmup(self.current_epoch, self.hparams.kl_weight_post, 5, 25)

        total = torch.stack([v for v in comps.values()]).sum()
        
        for name, val in comps.items():
            self.log(f"{stage}/loss/{name}", val, on_step=False, on_epoch=True, sync_dist=True)

        self.log(
            f"{stage}/loss_total", total,
            on_step=False, on_epoch=True, sync_dist=True
        )
        
        return {"loss": total}

    def training_step(self, batch, batch_idx):
                
        # ----- collect a few pairs for visuals this epoch -----
        if ((self.current_epoch + 1) % self.log_images_every_n_epochs == 0
            and len(self._val_vis_pairs) < self.vis_num_pairs):
            
            inputs, y_true, pmasks, rmasks = batch
            inputs = inputs.float().permute(0, 4, 1, 2, 3)
            y_true = y_true.float().permute(0, 4, 1, 2, 3)
            pmasks = pmasks.to(torch.int64).permute(0, 4, 1, 2, 3)
            rmasks = rmasks.to(torch.int64).permute(0, 4, 1, 2, 3)

            y_pred = self.model(inputs, y_true, pmasks, rmasks)

            with torch.no_grad():
                self._collect_pairs(inputs, y_true, y_pred) 
        
        return self._step(batch, stage="train")["loss"]

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def on_train_epoch_end(self):
        # push panels if this is a logging epoch
        if (self.current_epoch + 1) % self.log_images_every_n_epochs == 0:
            self._log_pairs_to_wandb()
        else:
            # clear any leftover if we decided not to log this epoch
            self._val_vis_pairs.clear()

    # ------------------------------ Optimizer --------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ------------------------------- Inference --------------------------------

    def forward(self, inputs, y_true, pmasks, rmasks):
        return self.model(inputs, y_true, pmasks, rmasks)
