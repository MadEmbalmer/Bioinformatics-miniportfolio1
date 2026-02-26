"""
Diffusion model for RNA torsion-angle generation.

This module defines:
- A cosine/cyclic-aware loss helper for angles
- A stable diffusion noise schedule (betas/alphas/alpha_bar)
- Forward noising (q_sample)
- An EGNN-based denoiser that predicts torsions given features + positions + time
- A simple reverse sampling loop (DDPM-style)

Notes
-----
- This is written to be data-agnostic: it does not bundle Kaggle data.
- It predicts torsions (e.g., alpha/beta) in degrees by default.
- Use an internal-coordinate decoder (utils/internal_coordinate_decoder.py)
  to convert torsions -> 3D coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.egnn import EGNN, EGNNConfig, TorsionHead


# -------------------------
# Noise schedule
# -------------------------

@dataclass
class DiffusionSchedule:
    """
    Precomputed diffusion schedule tensors.

    All tensors are float32 and live on the device you move them to.
    """
    betas: torch.Tensor           # [T]
    alphas: torch.Tensor          # [T]
    alpha_bars: torch.Tensor      # [T] cumulative product of alphas
    sqrt_alpha_bars: torch.Tensor # [T]
    sqrt_one_minus_alpha_bars: torch.Tensor # [T]


def make_linear_beta_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: Optional[torch.device] = None,
) -> DiffusionSchedule:
    """
    Classic linear beta schedule (DDPM).
    """
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def _extract(schedule_tensor: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract schedule values at timesteps t and reshape for broadcasting.

    schedule_tensor: [T]
    t: [B] integer timesteps
    returns: [B, 1, 1] or broadcastable to x_shape
    """
    # gather -> [B]
    out = schedule_tensor.gather(0, t)
    # reshape to [B, 1, 1]... to broadcast across sequence length and channels
    while out.ndim < len(x_shape):
        out = out.unsqueeze(-1)
    return out


# -------------------------
# Angle-aware losses
# -------------------------

def cyclic_mse_degrees(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """
    Cyclic MSE for angles in degrees.

    Equivalent to: mean(1 - cos(pred-target)), which respects periodicity.
    """
    diff = torch.deg2rad(pred_deg - target_deg)
    return torch.mean(1.0 - torch.cos(diff))


def smoothness_l2(pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Simple smoothness regularizer along sequence dimension.

    pred: [B, L, C]
    mask: [B, L] bool (optional)
    """
    if pred.size(1) < 2:
        return pred.new_tensor(0.0)

    delta = pred[:, 1:] - pred[:, :-1]  # [B, L-1, C]
    if mask is None:
        return torch.mean(delta ** 2)

    m = mask[:, 1:] & mask[:, :-1]  # [B, L-1]
    if m.sum() == 0:
        return pred.new_tensor(0.0)
    return torch.mean((delta[m]) ** 2)


# -------------------------
# Forward diffusion (q)
# -------------------------

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward noising step:
      x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps

    x0: [B, L, C]
    t: [B] int timesteps
    returns: (x_t, eps)
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab = _extract(schedule.sqrt_alpha_bars, t, x0.shape)
    sqrt_1mab = _extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape)

    xt = sqrt_ab * x0 + sqrt_1mab * noise
    return xt, noise


# -------------------------
# EGNN Diffusion Denoiser
# -------------------------

class EGNNTimeDenoiser(nn.Module):
    """
    Denoiser model: predicts x0 (torsions) from noised torsions + conditioning features.

    We condition on:
      - per-residue features: feats [B, L, F]
      - current positions: pos [B, L, 3]  (often random init or carried in sampling)
      - timestep embedding: scalar t in [0,1] appended to node features

    Output:
      - predicted torsions in degrees: [B, L, 2]
    """

    def __init__(
        self,
        feature_dim: int,
        egnn_cfg: EGNNConfig = EGNNConfig(),
        torsion_dim: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.torsion_dim = torsion_dim

        # We concatenate features + noised torsions + time scalar
        self.in_dim = feature_dim + torsion_dim + 1

        self.egnn = EGNN(in_dim=self.in_dim, cfg=egnn_cfg)
        self.head = TorsionHead(hidden_dim=egnn_cfg.hidden_dim, out_dim=torsion_dim)

    def forward(
        self,
        feats: torch.Tensor,          # [B, L, F]
        pos: torch.Tensor,            # [B, L, 3]
        edge_index: torch.Tensor,     # [2, E] over flattened nodes
        x_t: torch.Tensor,            # [B, L, torsion_dim] noised torsions
        t: torch.Tensor,              # [B] int timesteps
        T: int,
        mask: Optional[torch.Tensor] = None,  # [B, L] bool
    ) -> torch.Tensor:
        B, L, _ = feats.shape

        # time in [0,1]
        t_norm = (t.float() / float(T)).clamp(0.0, 1.0)  # [B]
        t_feat = t_norm[:, None, None].expand(B, L, 1)   # [B,L,1]

        x_in = torch.cat([feats, x_t, t_feat], dim=-1)   # [B,L,F+torsion+1]

        # Flatten for EGNN (expects [N,F], [N,3])
        x_flat = x_in.reshape(B * L, -1)
        pos_flat = pos.reshape(B * L, 3)

        h, _ = self.egnn(x_flat, pos_flat, edge_index)
        pred = self.head(h).reshape(B, L, self.torsion_dim)

        if mask is not None:
            pred = pred * mask.unsqueeze(-1)

        return pred


# -------------------------
# Reverse sampling (p)
# -------------------------

@torch.no_grad()
def p_sample_ddpm(
    model: EGNNTimeDenoiser,
    feats: torch.Tensor,        # [B,L,F]
    pos: torch.Tensor,          # [B,L,3]
    edge_index: torch.Tensor,   # [2,E]
    x_t: torch.Tensor,          # [B,L,C]
    t: torch.Tensor,            # [B] int
    schedule: DiffusionSchedule,
    T: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    One reverse DDPM step, producing x_{t-1} from x_t.

    We use a simple "predict x0" parameterization:
      pred_x0 = model(...)
      x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_pred) + sigma_t * z

    where eps_pred is derived from pred_x0 and x_t.
    """
    betas_t = _extract(schedule.betas, t, x_t.shape)
    alphas_t = _extract(schedule.alphas, t, x_t.shape)
    alpha_bar_t = _extract(schedule.alpha_bars, t, x_t.shape)

    # Predict x0 (torsions)
    pred_x0 = model(feats, pos, edge_index, x_t=x_t, t=t, T=T, mask=mask)

    # Derive eps prediction from x_t and pred_x0:
    # x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
    eps_pred = (x_t - torch.sqrt(alpha_bar_t) * pred_x0) / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)

    # DDPM mean
    mean = (1.0 / torch.sqrt(alphas_t + 1e-8)) * (x_t - (betas_t * eps_pred) / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8))

    # noise for stochasticity (skip when t==0)
    z = torch.randn_like(x_t)
    sigma = torch.sqrt(betas_t)  # simple choice
    nonzero = (t != 0).float()
    while nonzero.ndim < mean.ndim:
        nonzero = nonzero.unsqueeze(-1)

    x_prev = mean + nonzero * sigma * z

    if mask is not None:
        x_prev = x_prev * mask.unsqueeze(-1)

    return x_prev


@torch.no_grad()
def sample_torsions(
    model: EGNNTimeDenoiser,
    feats: torch.Tensor,            # [B,L,F]
    pos: torch.Tensor,              # [B,L,3]
    edge_index: torch.Tensor,       # [2,E]
    schedule: DiffusionSchedule,
    T: int = 1000,
    steps: int = 200,
    mask: Optional[torch.Tensor] = None,
    clamp_degrees: float = 180.0,
) -> torch.Tensor:
    """
    Generate torsions by reverse diffusion starting from Gaussian noise.

    Returns:
      torsions_deg: [B,L,2] (in degrees), optionally clamped.
    """
    device = feats.device
    B, L, _ = feats.shape
    C = model.torsion_dim

    x = torch.randn((B, L, C), device=device)

    # choose a subset of timesteps if steps < T
    # map step index -> diffusion timestep
    # e.g. steps=200 over T=1000 => stride=5
    stride = max(1, T // steps)
    timesteps = list(range(0, T, stride))
    if timesteps[-1] != T - 1:
        timesteps.append(T - 1)

    for t_int in reversed(timesteps):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        x = p_sample_ddpm(model, feats, pos, edge_index, x, t, schedule, T=T, mask=mask)

    if clamp_degrees is not None:
        x = torch.clamp(x, -clamp_degrees, clamp_degrees)

    return x