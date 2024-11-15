import torch
from torch.nn import functional as F
from torch import sin, cos, atan2, acos
import os
import numpy as np
import math
import random


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        if self.betas.device != t_int.device:
            self.betas = self.betas.to(t_int.device)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        if self.alphas_bar.device != t_int.device:
            self.alphas_bar = self.alphas_bar.to(t_int.device)
        return self.alphas_bar[t_int.long()]


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def seq_recovery(data, pred_seq):
    '''
    data.x is nature sequence

    '''
    ind = (data.x.argmax(dim=1) == pred_seq.argmax(dim=1))
    recovery = ind.sum() / ind.shape[0]
    return recovery, ind.cpu()


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def set_seed(seed=1024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def place_fourth_atom(
        a_coord: torch.Tensor,
        b_coord: torch.Tensor,
        c_coord: torch.Tensor,
        length: torch.Tensor,
        planar: torch.Tensor,
        dihedral: torch.Tensor,
) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])

    return d_coord


def place_missing_cb(atom_positions):
    cb_coords = place_fourth_atom(atom_positions[:, 2], atom_positions[:, 0],
                                  atom_positions[:, 1], torch.tensor(1.522),
                                  torch.tensor(1.927), torch.tensor(-2.143))
    cb_coords = torch.where(torch.isnan(cb_coords), torch.zeros_like(cb_coords), cb_coords)

    # replace all vitural cb coords
    atom_positions[:, 3] = cb_coords
    return atom_positions


def place_missing_o(atom_positions, missing_mask):
    o_coords = place_fourth_atom(
        torch.roll(atom_positions[:, 0], shifts=-1, dims=0), atom_positions[:, 1],
        atom_positions[:, 2], torch.tensor(1.231), torch.tensor(2.108),
        torch.tensor(-3.142))
    o_coords = torch.where(torch.isnan(o_coords), torch.zeros_like(o_coords), o_coords)

    atom_positions[:, 4][missing_mask == 0] = o_coords[missing_mask == 0]
    return atom_positions


def cal_stats_metric(metric_list):
    mean_metric = np.mean(metric_list)
    median_metric = np.median(metric_list)
    return mean_metric, median_metric


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_entropy(log_probs):
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -1 * p_log_p.mean(dim=-1)
    return entropy


def fuse_logits_by_log_probs(log_prob_list, logits_list, temp=1.):
    entropy_list = [get_entropy(log_probs) for log_probs in log_prob_list]
    entropy = torch.stack(entropy_list, dim=0)
    entropy = torch.nn.functional.softmax(-1 * entropy / temp, dim=0)

    # fuse by entropy
    logits_list = torch.stack(logits_list, dim=0)
    logits = (entropy.unsqueeze(-1) * logits_list).sum(dim=0)

    return logits


def sin_mask_ratio_adapter(beta_t_bar, max_deviation=0.2, center=0.5):
    adjusted = beta_t_bar * torch.pi * 0.5
    sine = torch.sin(adjusted)
    adjustment = sine * max_deviation
    mask_ratio = center + adjustment
    return mask_ratio.squeeze(1)
