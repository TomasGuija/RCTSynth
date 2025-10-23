import torch
import torch.nn.functional as F
import torch


@torch.no_grad()
def grad_mae_3d(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Gradient-magnitude MAE over the last three spatial dims.
    """
    assert y_true.shape == y_pred.shape and y_true.dim() >= 3

    # Gradients along last 3 dims
    grads_t = torch.gradient(y_true, dim=(-3, -2, -1))
    grads_p = torch.gradient(y_pred, dim=(-3, -2, -1))

    # Gradient magnitude
    g_t = torch.sqrt(sum(g**2 for g in grads_t))
    g_p = torch.sqrt(sum(g**2 for g in grads_p))

    return (g_t - g_p).abs().mean()
