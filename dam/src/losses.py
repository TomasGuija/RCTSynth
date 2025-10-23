import torch
import torch.nn.functional as F
import numpy as np
import math


def _per_sample_percentile_threshold(
    x: torch.Tensor,          # (B,1,X,Y,Z)
    p_air: float = 0.08,      # lowest p% treated as air
) -> torch.Tensor:
    assert x.dim() == 5 and x.size(1) == 1, "expected (B,1,X,Y,Z)"
    B = x.shape[0]
    flat = x.reshape(B, -1)

    # clamp p_air and compute common k for all samples (same voxel count)
    p = float(max(0.0, min(1.0, p_air)))
    N = flat.size(1)
    k = max(1, min(N, int(round(p * N))))  # 1..N

    # kth *smallest* value per sample (vectorized)
    # torch.kthvalue returns (values, indices)
    thr_vals, _ = torch.kthvalue(flat, k=k, dim=1, keepdim=True)  # (B,1)
    thr = thr_vals.view(B, 1, 1, 1, 1).to(dtype=x.dtype, device=x.device)

    return thr

#TODO: find a better way to do this, more robustly
@torch.no_grad()
def build_anatomy_weights(
    ct_rep: torch.Tensor,   # (B,1,X[,Y[,Z]])
    *,
    p_air: float = 0.08,    # percentile to separate air/body
) -> torch.Tensor:
    """
    Hard mask: 1 inside body (ct_rep > thr), 0 elsewhere.
    """
    thr = _per_sample_percentile_threshold(ct_rep, p_air=p_air)
    M = (ct_rep > thr).float()
    return M.clamp_min(1e-6)  # keep tiny >0 to avoid zero-mass issues if needed


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(
        self,
        win=None,
        device='cpu',
        use_weights: bool = False,
        # params forwarded to build_anatomy_weights when use_weights=True:
        p_air: float = 0.08,
    ):
        self.win = win
        self.device = device
        self.use_weights = use_weights
        self._w_params = dict(
            p_air=p_air,
        )

    def loss(self, y_true, y_pred, weight: torch.Tensor | None = None, eps: float = 1e-5):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device, dtype=Ii.dtype)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # build weights if requested and not provided
        if weight is None and self.use_weights:
            W = build_anatomy_weights(Ii, **self._w_params).to(Ii.dtype)
        else:
            W = None if weight is None else weight.to(Ii.dtype)

        if W is None:
            # compute CC squares
            I2 = Ii * Ii
            J2 = Ji * Ji
            IJ = Ii * Ji

            I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            win_size = np.prod(win)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        else:
            # --- Weighted path: proper weighted local NCC ---
            # Per-window mass (sum of weights)
            W_sum = conv_fn(W, sum_filt, stride=stride, padding=padding).clamp_min(1e-6)

            I_sum  = conv_fn(Ii * W,  sum_filt, stride=stride, padding=padding)
            J_sum  = conv_fn(Ji * W,  sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn((Ii * Ii) * W, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn((Ji * Ji) * W, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn((Ii * Ji) * W, sum_filt, stride=stride, padding=padding)

            u_I = I_sum / W_sum
            u_J = J_sum / W_sum

            # Covariance and variances with weights
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + (u_I * u_J) * W_sum
            I_var = I2_sum - 2 * u_I * I_sum + (u_I * u_I) * W_sum
            J_var = J2_sum - 2 * u_J * J_sum + (u_J * u_J) * W_sum

        cc = cross * cross / (I_var * J_var + eps)

        if W is None:
            cc_mean = cc.mean()
        else:
            # valid = (W_sum > 1e-6).to(cc.dtype)
            # cc_mean = (cc * valid).sum() / valid.sum().clamp_min(1.0)
            cc_mean = (cc * W_sum).sum() / W_sum.sum().clamp_min(1.0)

        return 1-cc_mean


class MSE:
    """
    Mean squared error loss.
    """

    def __init__(
        self,
        sqrt: bool = False,
        use_weights: bool = False,
        # Parameters for build_anatomy_weights
        p_air: float = 0.08,
    ):
        self.sqrt = sqrt
        self.use_weights = use_weights
        self._w_params = dict(
            p_air=p_air,
        )

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6):
        diff = (y_true - y_pred) ** 2

        # Build weights if requested and not provided
        if weight is None and self.use_weights:
            W = build_anatomy_weights(y_true, **self._w_params).to(y_true.dtype)
        else:
            W = None if weight is None else weight.to(y_true.dtype)

        if W is None:
            mse = diff.mean()
        else:
            num = (W * diff).sum()
            den = W.sum().clamp_min(eps)
            mse = num / den

        return torch.sqrt(mse) if self.sqrt else mse


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1-dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class KL:
    """
    Kullback–Leibler divergence between 2 Gaussian distributions.
    """

    def __init__(self, prior=True):
        self.prior = prior

    def loss(self, _, param):
        """
        Parameters:
            param: list with means and log variances
        """
        
        if self.prior: 
            p = torch.distributions.multivariate_normal.MultivariateNormal(
                param['p_mu'],
                torch.diag_embed(torch.exp(param['p_logvar']))
                )
            q = torch.distributions.multivariate_normal.MultivariateNormal(
                param['q_mu'],
                torch.diag_embed(torch.exp(param['q_logvar']))
                )
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(q, p))
        else:
            kl_div = 1 + param['q_logvar'] - torch.square(param['q_mu']) - torch.exp(param['q_logvar'])
            kl_loss = torch.mean(-0.5 * torch.sum(kl_div, dim=1), dim=0)

        return kl_loss