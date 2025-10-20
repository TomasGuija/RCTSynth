from typing import Any, Dict, List, Tuple, Optional
import torch
from lightning.pytorch import LightningModule
import wandb
from lightning.pytorch.loggers import WandbLogger

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
        kl_weight: float,
        bidir: bool,
        latent_dim: int,
        enc_nf: List[int],
        dec_nf: List[int],
        int_steps: int,
        int_downsize: int,
        conv_per_level: int,
        prior: str,
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
        self._loss_ready = False
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

    def _pick_slice(self, vol: torch.Tensor) -> torch.Tensor:
        """
        vol: [B, C, X, Y, Z]
        returns: [B, X, Y] selected slice or MIP
        """
        c = self.vis_channel
        volc = vol[:, c]  # [B, X, Y, Z]

        if self.vis_slice_mode == "mip":
            # max-intensity projection along Z
            img = volc.max(dim=-1).values  # [B, X, Y]
        else:
            # mid or index
            Z = volc.shape[-1]
            z = Z // 2 if (self.vis_slice_mode == "mid" or self.vis_slice_index is None) else int(self.vis_slice_index)
            z = max(0, min(Z - 1, z))
            img = volc[..., z]  # [B, X, Y]
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

        gt2d = self._normalize01(self._pick_slice(y_true)).detach().cpu()
        rc2d = self._normalize01(self._pick_slice(recon)).detach().cpu()

        # how many more do we need?
        remaining = self.vis_num_pairs - len(self._val_vis_pairs)
        take = max(0, min(remaining, gt2d.shape[0]))
        for i in range(take):
            # each item is (GT, Recon) as [X, Y] tensors
            self._val_vis_pairs.append((gt2d[i], rc2d[i]))

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

        panels = []
        for idx, (gt, rc) in enumerate(self._val_vis_pairs):
            # side-by-side concat along width: [X, 2Y]
            concat = torch.cat([gt, rc], dim=1)  # (X,Y) cat width -> dim=1
            panels.append(wandb.Image(concat.numpy(), caption=f"pair {idx}: GT | Recon"))

        wb.experiment.log({ "val/examples_gt_vs_recon": panels, "epoch": self.current_epoch })
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

        if not self._loss_ready:
            if self._image_loss_name == "ncc":
                img_loss = dam.losses.NCC(device=self.device).loss
            elif self._image_loss_name == "mse":
                img_loss = dam.losses.MSE().loss
            else:
                raise ValueError(f"Invalid image_loss: {self._image_loss_name}")

            self.loss_fns: List[Any] = []
            self.loss_wts: List[float] = []

            if self.hparams.bidir:
                self.loss_fns += [img_loss, img_loss]
                self.loss_wts += [0.5 * self.hparams.lambda_img, 0.5 * self.hparams.lambda_img]
                self.loss_fns += [dam.losses.Dice().loss, dam.losses.Dice().loss]
                self.loss_wts += [self.hparams.kappa, self.hparams.kappa]
            else:
                self.loss_fns += [img_loss]
                self.loss_wts += [self.hparams.lambda_img]
                self.loss_fns += [dam.losses.Dice().loss]
                self.loss_wts += [self.hparams.kappa]

            self.loss_fns += [dam.losses.Grad("l2", loss_mult=self.hparams.int_downsize).loss]
            self.loss_wts += [self.hparams.beta]

            self.loss_fns += [dam.losses.KL(prior=self.hparams.prior).loss]
            self.loss_wts += [self.hparams.kl_weight]

            self._loss_ready = True

            self.loss_names: List[str] = []
            if self.hparams.bidir:
                # name them clearly; tweak to your taste
                self.loss_names += ["img_fwd", "img_bwd"]
                self.loss_names += ["dice_fwd", "dice_bwd"]
            else:
                self.loss_names += ["img"]
                self.loss_names += ["dice"]
            self.loss_names += ["grad", "kl"]

            assert len(self.loss_fns) == len(self.loss_wts) == len(self.loss_names)

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

        # one-hot masks
        pmasks = self.model.to_binary(pmasks, num_organs=self.num_organs)
        rmasks = self.model.to_binary(rmasks, num_organs=self.num_organs)

        # targets for each loss
        if self.bidir:
            targets = [y_true, inputs, rmasks, pmasks, 0, 0]
        else:
            targets = [y_true, rmasks, 0, 0]

        total = 0.0
        for i, (loss_fn, wt, name) in enumerate(zip(self.loss_fns, self.loss_wts, self.loss_names)):
            tgt_i = targets[i]
            pred_i = y_pred[i]  # keep the same ordering as in your original training loop
            comp = loss_fn(tgt_i, pred_i) * wt
            total = total + comp

            # per-component logging with human-readable names
            self.log(
                f"{stage}/loss/{name}", comp,
                on_step=False, on_epoch=True, sync_dist=True
            )

        # total loss (used for backprop when stage=="train")
        self.log(
            f"{stage}/loss_total", total,
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        return {"loss": total}

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")["loss"]

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

        inputs, y_true, pmasks, rmasks = batch
        inputs = inputs.float().permute(0, 4, 1, 2, 3)
        y_true = y_true.float().permute(0, 4, 1, 2, 3)
        pmasks = pmasks.to(torch.int64).permute(0, 4, 1, 2, 3)
        rmasks = rmasks.to(torch.int64).permute(0, 4, 1, 2, 3)

        y_pred = self.model(inputs, y_true, pmasks, rmasks)

        # ----- collect a few pairs for visuals this epoch -----
        if ((self.current_epoch + 1) % self.log_images_every_n_epochs == 0
            and len(self._val_vis_pairs) < self.vis_num_pairs):
            with torch.no_grad():
                self._collect_pairs(inputs, y_true, y_pred)

    def on_validation_epoch_end(self):
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
