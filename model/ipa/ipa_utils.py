import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import importlib
from scipy.stats import truncnorm
from typing import Optional, Callable, List


def cal_dihedrals(X, eps=1e-7):
    """
    Compute dihedral angles from a set of coordinates.
    """
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    D = F.pad(D, (1, 2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    phi, psi, omega = torch.unbind(D, -1)
    D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    return D_features  # (B, L, 6), the 6 is cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)


def cal_pair_rbf(pairwise_distance, dist_bins=24, dist_bin_width=0.5):
    device = pairwise_distance.device
    D_min, D_max, D_count = 0., dist_bins * dist_bin_width, dist_bins
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(pairwise_distance, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def relative_pairwise_position_idx(seq_len, rel_pos_k=32):
    indices = torch.arange(seq_len, dtype=torch.long)
    indices = indices[:, None] - indices[None, :]
    indices = indices.clamp(-rel_pos_k, rel_pos_k)
    indices = indices + rel_pos_k
    return indices


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def is_fp16_enabled():
    # Autocast world
    try:
        fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
        fp16_enabled = fp16_enabled and torch.is_autocast_enabled()
    except AttributeError:
        fp16_enabled = False

    return fp16_enabled


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: str = "default",
            init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out
