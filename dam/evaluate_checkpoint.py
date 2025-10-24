import argparse
import os
from typing import List, Tuple, Optional
import numpy as np
import torch
import zipfile
import traceback

from dam.lightning.dam_datamodule import DamDataModule
from dam.lightning.dam_lightning_module import DamLightning
from dam.utils.elastix import elastix_register_and_resample
from dam.src.metrics import grad_mae_3d

def ncc_torch(gt: torch.Tensor, img: torch.Tensor) -> float:
    g = gt - gt.mean()
    r = img - img.mean()
    denom = (g.square().sum().sqrt() * r.square().sum().sqrt()) + 1e-12
    return float((g * r).sum() / denom)

def ncc_numpy(gt: np.ndarray, img: np.ndarray) -> float:
    g = gt - gt.mean()
    r = img - img.mean()
    denom = np.sqrt((g ** 2).sum() * (r ** 2).sum()) + 1e-12
    return float((g * r).sum() / denom)

def ensure_same_shape(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    # try permutations
    for perm in [(0,1,2),(2,1,0),(0,2,1),(1,0,2),(1,2,0),(2,0,1)]:
        t = np.transpose(arr, perm)
        if t.shape == target_shape:
            return t
    # fallback: try resize via simple interpolation (not ideal) -> just reshape if same number of voxels
    if arr.size == np.prod(target_shape):
        return arr.reshape(target_shape)
    # if nothing matches, raise for visibility
    raise ValueError(f"Could not match elastix output shape {arr.shape} to target {target_shape}")

def evaluate(
    checkpoint: str,
    dataset_path: str,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "cuda",
    eval_n: Optional[int] = None,
    run_ncc: bool = True,
    run_grad: bool = True,
    run_elastix: bool = True,
    save_path: Optional[str] = None,
    wandb_log: bool = False,
):
    # Build datamodule (defaults aligned with your config.yaml)
    dm = DamDataModule(
        dataset_path=dataset_path,
        xkey="planning",
        ykey="repeated",
        xmask="masks_planning",
        ymask="masks_repeated",
        xamask="planning_anatomy_mask",
        yamask="repeated_anatomy_mask",
        maxv=1.0,
        minv=0.0,
        train_split=0.95,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        enable_swapping=False,
    )
    dm.setup()
    map_loc = torch.device(device)

    ckpt = torch.load(checkpoint, map_location=map_loc)

    val_loader = dm.val_dataloader()
    sd = ckpt["state_dict"]

    # sd_remapped = {rename(k): v for k, v in sd.items()}
    hparams = ckpt.get("hyper_parameters")

    model = DamLightning(**hparams)
    model.build_model(dm.inshape)
    # Load model checkpoint
    model.load_state_dict(sd)

    # Ensure inshape/hyperparams are set as in your datamodule and initialize internal components
    model.hparams.inshape = dm.inshape
    model.to(map_loc)
    model.eval()

    device_t = map_loc

    # Accumulators
    model_nccs: List[float] = []
    grad_maes: List[float] = []
    elx_nccs: List[float] = []
    diffs: List[float] = []

    processed = 0
    with torch.no_grad():
        for batch in val_loader:
            # prepare batch (permutes and casts)
            b = model._prepare_batch(batch)
            inputs = b["inputs"].to(device_t)   # [B, C, X, Y, Z]
            y_true = b["y_true"].to(device_t)   # [B, 1, X, Y, Z]
            pmasks = b["pmasks"].to(device_t)
            rmasks = b["rmasks"].to(device_t)

            # run model
            y_pred = model.forward(inputs, y_true, pmasks, rmasks)
            pred = model._model_outputs_to_dict(y_pred, model.bidir)
            recon = pred["rep"].detach().cpu()   # [B,1,X,Y,Z] on CPU

            B = recon.shape[0]
            for i in range(B):
                if eval_n is not None and processed >= eval_n:
                    break

                gt = y_true[i:i+1].cpu().squeeze(0).squeeze(0)
                rc = recon[i:i+1].squeeze(0).squeeze(0)

                # model NCC
                if run_ncc:
                    try:
                        mncc = ncc_torch(gt, rc)
                    except Exception:
                        # fallback: convert to numpy
                        mncc = ncc_numpy(gt.numpy(), rc.numpy())
                    model_nccs.append(mncc)

                # grad MAE
                if run_grad:
                    # grad_mae_3d expects full tensors with batch dim? it expects shapes >=3 and computes gradient over last 3 dims
                    # Use single-sample tensors with shape [1,X,Y,Z] to fit expectations
                    gt_t = gt.unsqueeze(0)   # [1,X,Y,Z]
                    rc_t = rc.unsqueeze(0)
                    gmae = float(grad_mae_3d(gt_t, rc_t).cpu().numpy())
                    grad_maes.append(gmae)

                # Elastix comparison
                if run_elastix:
                    # Inputs for elastix are numpy volumes in [X,Y,Z]
                    planning_np = inputs[i:i+1, 0].squeeze(0).cpu().numpy()  # pick first channel [X,Y,Z]
                    repeat_np = gt.cpu().numpy()                  # [X,Y,Z]

                    try:
                        reg_np = elastix_register_and_resample(planning_np, repeat_np)
                        # Ensure shape matches
                        reg_np = ensure_same_shape(reg_np, repeat_np.shape)
                    except Exception as e:
                        # If elastix fails, skip this sample with a warning
                        print(f"[warning] elastix failed for sample {processed}: {e}")
                        continue

                    encc = ncc_numpy(repeat_np, reg_np)
                    elx_nccs.append(encc)

                    # compute diff (model_ncc - elastix_ncc) if model_ncc is available
                    if run_ncc:
                        diff = mncc - encc
                        diffs.append(diff)

                processed += 1

            if eval_n is not None and processed >= eval_n:
                break

    # Aggregate
    results = {}
    if run_ncc and len(model_nccs) > 0:
        results["mean_model_ncc"] = float(np.mean(model_nccs))
    if run_grad and len(grad_maes) > 0:
        results["mean_grad_mae"] = float(np.mean(grad_maes))
    if run_elastix and len(elx_nccs) > 0:
        results["mean_elx_ncc"] = float(np.mean(elx_nccs))
    if run_elastix and run_ncc and len(diffs) > 0:
        results["mean_model_minus_elx_ncc"] = float(np.mean(diffs))

    print("Evaluation summary:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    # Save or plot
    if save_path or wandb_log:
        # Build example panels using same method as training
        panels = []
        panel_metrics = []
        
        # Process N samples (or all if eval_n is None)
        with torch.no_grad():
            for batch in val_loader:
                b = model._prepare_batch(batch)
                inputs = b["inputs"].to(device_t)
                y_true = b["y_true"].to(device_t)
                pmasks = b["pmasks"].to(device_t)
                rmasks = b["rmasks"].to(device_t)

                y_pred = model.forward(inputs, y_true, pmasks, rmasks)
                pred = model._model_outputs_to_dict(y_pred, model.bidir)
                
                # Process each sample in batch
                for i in range(inputs.size(0)):
                    if eval_n is not None and len(panels) >= eval_n:
                        break
                        
                    # Extract and normalize 2D slices using same methods as training
                    pl2d = model._normalize01(model._pick_slice(inputs[i:i+1]))[0].detach().cpu()
                    gt2d = model._normalize01(model._pick_slice(y_true[i:i+1]))[0].detach().cpu()
                    rc2d = model._normalize01(model._pick_slice(pred["rep"][i:i+1]))[0].detach().cpu()
                    
                    # Use same panel builder as training
                    panel, metrics = model._build_pair_panel(pl2d, gt2d, rc2d)
                    panels.append(panel)
                    panel_metrics.append(metrics)
                    
                if eval_n is not None and len(panels) >= eval_n:
                    break

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            # Save all panel images individually
            for i, panel in enumerate(panels):
                panel.image.save(os.path.join(save_path, f"sample_{i:03d}.png"))
            
            # Save metrics
            np.savez_compressed(
                os.path.join(save_path, "eval_results.npz"),
                model_nccs=np.array(model_nccs),
                grad_maes=np.array(grad_maes),
                elx_nccs=np.array(elx_nccs),
                diffs=np.array(diffs),
                panel_metrics=panel_metrics,
                summary=results,
            )

        if wandb_log:
            try:
                import wandb
                run = wandb.init(project="RCTSynth_eval", reinit=True)
                run.log(results)
                # Log panels in same format as training
                run.log({"examples_gt_vs_recon": panels})
                run.finish()
            except Exception as e:
                print(f"[warning] wandb logging failed: {e}")

    return results

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to Lightning checkpoint (.ckpt)")
    p.add_argument("--dataset", required=True, help="path to H5 dataset")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval_n", type=int, default=None, help="max number of samples to evaluate (None => all)")
    p.add_argument("--no_ncc", action="store_true", help="skip model NCC")
    p.add_argument("--no_grad", action="store_true", help="skip grad MAE")
    p.add_argument("--no_elastix", action="store_true", help="skip elastix comparison")
    p.add_argument("--save_path", type=str, default=None, help="where to save numeric results and plots")
    p.add_argument("--wandb", action="store_true", help="log summary and plots to wandb")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = evaluate(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        eval_n=args.eval_n,
        run_ncc=not args.no_ncc,
        run_grad=not args.no_grad,
        run_elastix=not args.no_elastix,
        save_path=args.save_path,
        wandb_log=args.wandb,
    )
    # concise exit
    print(res)