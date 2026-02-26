"""
EGNN (E(n)-Equivariant Graph Neural Network) components.

This module provides a small, reusable EGNN implementation for geometric
message passing with coordinate updates.

- Node features are updated via edge messages.
- Coordinates are updated equivariantly using relative coordinate differences.
- The model can be used as a building block inside diffusion or direct
  torsion/coordinate prediction pipelines.

Dependencies:
- torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class EGNNConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.0
    use_layer_norm: bool = True


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGNNLayer(nn.Module):
    """
    Single EGNN layer.

    Inputs:
      x:   [N, F] node features
      pos: [N, 3] node coordinates
      edge_index: [2, E] (row=source, col=target) indices

    Returns:
      x_out:   [N, F] updated node features
      pos_out: [N, 3] updated coordinates
    """

    def __init__(self, feat_dim: int, dropout: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        # Edge message uses: x_i, x_j, ||pos_i - pos_j||^2
        self.edge_mlp = MLP(in_dim=2 * feat_dim + 1, out_dim=feat_dim, hidden_dim=feat_dim, dropout=dropout)

        # Node update after aggregation
        self.node_mlp = MLP(in_dim=feat_dim, out_dim=feat_dim, hidden_dim=feat_dim, dropout=dropout)

        # Coordinate scaling from edge message -> scalar
        self.coord_mlp = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Tanh(),  # keeps coordinate updates bounded
        )

        self.norm = nn.LayerNorm(feat_dim) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or pos.ndim != 2 or pos.size(-1) != 3:
            raise ValueError("Expected x: [N,F], pos: [N,3].")

        row, col = edge_index  # [E], [E]
        # relative vectors and squared distances
        diff = pos[row] - pos[col]  # [E,3]
        dist2 = (diff * diff).sum(dim=1, keepdim=True)  # [E,1]

        # edge message
        edge_feat = torch.cat([x[row], x[col], dist2], dim=1)  # [E, 2F+1]
        msg = self.edge_mlp(edge_feat)  # [E, F]

        # aggregate messages to nodes (sum over incoming edges)
        agg = torch.zeros_like(x)  # [N,F]
        agg.index_add_(0, row, msg)

        # node update (residual + norm)
        x_out = self.norm(x + self.node_mlp(agg))

        # coordinate update: scalar(msg) * diff
        scale = self.coord_mlp(msg)  # [E,1]
        delta = torch.zeros_like(pos)  # [N,3]
        delta.index_add_(0, row, scale * diff)
        pos_out = pos + delta

        return x_out, pos_out


class EGNN(nn.Module):
    """
    Stacked EGNN.

    Typical usage:
      - Project input features to hidden_dim
      - Run EGNN layers
      - Output a prediction head (torsions, distances, etc.)

    Inputs:
      x: [N, in_dim]
      pos: [N, 3]
      edge_index: [2, E]

    Returns:
      h: [N, hidden_dim]
      pos: [N, 3]
    """

    def __init__(self, in_dim: int, cfg: EGNNConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(cfg.hidden_dim, dropout=cfg.dropout, use_layer_norm=cfg.use_layer_norm) for _ in range(cfg.num_layers)]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        return h, pos


class TorsionHead(nn.Module):
    """
    Simple head mapping node embeddings -> 2 torsion angles (e.g., alpha/beta).
    """

    def __init__(self, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)