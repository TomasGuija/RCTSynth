import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    Optionally masked by a binary mask (1=anatomy, 0=background).
    """
    def __init__(self, win=None, device='cpu'):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred, mask: torch.Tensor = None, eps: float = 1e-5):
        Ii = y_true
        Ji = y_pred

        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win
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

        conv_fn = getattr(F, 'conv%dd' % ndims)

        if mask is not None:
            # Convert mask to binary float (0/1)
            W = torch.clamp(mask.to(Ii.dtype), 0, 1)
            W_sum = conv_fn(W, sum_filt, stride=stride, padding=padding)
            
            # Only compute where mask is non-zero
            I_sum  = conv_fn(Ii * W,          sum_filt, stride=stride, padding=padding)
            J_sum  = conv_fn(Ji * W,          sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn((Ii * Ii) * W,   sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn((Ji * Ji) * W,   sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn((Ii * Ji) * W,   sum_filt, stride=stride, padding=padding)
            
            W_safe = W_sum.clamp_min(eps)
            u_I = I_sum / W_safe
            u_J = J_sum / W_safe

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + (u_I * u_J) * W_sum
            I_var = I2_sum - 2 * u_I * I_sum + (u_I * u_I) * W_sum
            J_var = J2_sum - 2 * u_J * J_sum + (u_J * u_J) * W_sum

            win_size = float(np.prod(win))
            valid = W_sum > 0

            cc = cross * cross / (I_var * J_var + eps)

            # Only average over valid regions
            cc = torch.where(valid, cc, torch.zeros_like(cc))
            valid_count = valid.float().sum().clamp_min(1.0)
            cc_mean = cc.sum() / valid_count
        else:
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

            cc = cross * cross / (I_var * J_var + eps)
            cc_mean = cc.mean()

        return 1 - cc_mean


class MSE:
    """
    Mean squared error loss with optional binary masking.
    """
    def __init__(self, sqrt: bool = False):
        self.sqrt = sqrt

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-6):
        diff = (y_true - y_pred) ** 2

        if mask is not None:
            # Binary mask
            W = (mask > 0).to(diff.dtype)
            # Average only over masked region
            mse = (diff * W).sum() / W.sum().clamp_min(eps)
        else:
            mse = diff.mean()

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