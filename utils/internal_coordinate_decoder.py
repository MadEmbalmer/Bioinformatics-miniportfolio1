"""
Internal coordinate decoder: torsion angles -> 3D coordinates (C1' atoms).

This module converts a sequence of torsion angles (e.g., alpha/beta in degrees)
into Cartesian coordinates using a simple internal-coordinate construction.

Notes
-----
- This is a lightweight geometric decoder meant for backbone-style coordinate
  reconstruction from predicted torsions.
- It uses Rodrigues' rotation formula to rotate a direction vector around a
  local normal defined by previous bonds.
- It does NOT enforce full RNA stereochemistry; it is a practical decoder for
  generating plausible coordinate traces for ML pipelines.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class InternalCoordinateDecoder(nn.Module):
    """
    Convert torsion angles to 3D coordinates.

    Parameters
    ----------
    bond_length : float
        Approximate C1'-C1' step length in Angstrom.
    """

    def __init__(self, bond_length: float = 1.6):
        super().__init__()
        self.bond_length = float(bond_length)

    def forward(self, torsions_deg: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        torsions_deg : torch.Tensor
            Shape [B, L, 2]. Two torsion angles per residue, in degrees.
        mask : Optional[torch.Tensor]
            Shape [B, L] boolean mask of valid residues. If None, all residues are valid.

        Returns
        -------
        coords : torch.Tensor
            Shape [B, L, 3] of reconstructed coordinates.
        """
        if torsions_deg.ndim != 3 or torsions_deg.size(-1) != 2:
            raise ValueError("torsions_deg must have shape [B, L, 2]")

        B, L, _ = torsions_deg.shape
        device = torsions_deg.device
        dtype = torsions_deg.dtype

        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            if mask.shape != (B, L):
                raise ValueError(f"mask must have shape [B, L] = {(B, L)}")
            mask = mask.to(device=device)

        coords = torch.zeros((B, L, 3), dtype=dtype, device=device)

        # Need at least 3 points to define a plane robustly.
        # Initialize first 3 residues in a fixed configuration.
        if L >= 1:
            coords[:, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
        if L >= 2:
            coords[:, 1] = torch.tensor([self.bond_length, 0.0, 0.0], dtype=dtype, device=device)
        if L >= 3:
            coords[:, 2] = torch.tensor([self.bond_length, self.bond_length, 0.0], dtype=dtype, device=device)

        # Build remaining coordinates using a rotating direction vector.
        for i in range(3, L):
            # previous three points
            p0 = coords[:, i - 3]
            p1 = coords[:, i - 2]
            p2 = coords[:, i - 1]

            # bond directions (normalized)
            b1 = p0 - p1
            b2 = p1 - p2
            b1 = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-8)
            b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)

            # normal to plane formed by b1 and b2
            n = torch.cross(b1, b2, dim=-1)
            n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)

            alpha = torch.deg2rad(torsions_deg[:, i, 0])
            cos_a = torch.cos(alpha).unsqueeze(-1)
            sin_a = torch.sin(alpha).unsqueeze(-1)

            # Rodrigues' rotation formula: rotate b2 around axis n by angle alpha
            new_dir = (
                b2 * cos_a
                + torch.cross(n, b2, dim=-1) * sin_a
                + n * (torch.sum(n * b2, dim=-1, keepdim=True)) * (1.0 - cos_a)
            )

            coords[:, i] = coords[:, i - 1] + new_dir * self.bond_length

        # Apply mask (invalid residues -> 0 coords)
        coords = coords * mask.unsqueeze(-1).to(dtype=dtype)
        return coords


# Convenience function
_decoder = InternalCoordinateDecoder()


def torsion_to_coords(torsions_deg: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Convenience wrapper.

    Accepts:
      - [L, 2] -> returns [L, 3]
      - [B, L, 2] -> returns [B, L, 3]

    Returns torch.Tensor (not numpy) to keep everything in PyTorch.
    """
    if torsions_deg.ndim == 2:
        torsions_deg = torsions_deg.unsqueeze(0)  # [1, L, 2]
        coords = _decoder(torsions_deg, mask=None if mask is None else mask.unsqueeze(0))
        return coords.squeeze(0)
    if torsions_deg.ndim == 3:
        return _decoder(torsions_deg, mask=mask)
    raise ValueError("torsions_deg must have shape [L,2] or [B,L,2]")