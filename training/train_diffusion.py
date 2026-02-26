import os, textwrap, json, pathlib

code = r'''#!/usr/bin/env python3
"""
Train an EGNN-based diffusion denoiser for RNA torsion angles.

This script is a clean, portfolio-friendly entry point that wraps the core
model code in `models/diffusion_model.py`.

Expected inputs (not bundled with this repo):
- A per-residue feature table (e.g., features/final_merged_features.csv)
  containing at least: target_id, resid, resname, and numeric feature columns.
- A torsion label file extracted from structures (e.g., features/torsion_labels.csv)
  containing at least: ID (target_id_resid), alpha, beta (in degrees).

Outputs:
- checkpoints/diffusion_denoiser.pt  (model weights)
- checkpoints/train_config.json      (training config)

Example:
    python training/train_diffusion.py \
        --features features/final_merged_features.csv \
        --torsions features/torsion_labels.csv \
        --outdir checkpoints \
        --epochs 20 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# torch-geometric is used ONLY to build radius graphs
try:
    from torch_geometric.nn import radius_graph
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for radius_graph. "
        "Install with: pip install torch-geometric"
    ) from e

# Make project root importable so `models.*` works when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion_model import (  # noqa: E402
    EGNNTimeDenoiser,
    make_linear_beta_schedule,
    q_sample,
    cyclic_mse_degrees,
    smoothness_l2,
)
from models.egnn import EGNNConfig  # noqa: E402


# -------------------------
# Dataset
# -------------------------

ID_COLS = ["target_id", "resid", "resname"]


def _parse_id_to_target_resid(id_str: str) -> Tuple[str, int]:
    """
    Kaggle-style ID is often: {target_id}_{resid}.
    target_id itself can contain underscores, so split on last underscore.
    """
    s = str(id_str)
    if "_" not in s:
        raise ValueError(f"Unexpected ID format (no underscore): {s}")
    target, resid = s.rsplit("_", 1)
    return target, int(resid)


class RNATorsionFeatureDataset(Dataset):
    """
    Groups per-residue features by target_id and aligns them to torsion labels.

    Returns variable-length sequences; DataLoader uses a custom collate_fn.
    """

    def __init__(
        self,
        feature_csv: str,
        torsion_csv: str,
        feature_drop: Optional[List[str]] = None,
        max_len: Optional[int] = None,
        use_versions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.feature_csv = feature_csv
        self.torsion_csv = torsion_csv
        self.feature_drop = set(feature_drop or [])
        self.max_len = max_len
        self.use_versions = set(use_versions) if use_versions else None

        feat_df = pd.read_csv(feature_csv, low_memory=False)
        tor_df = pd.read_csv(torsion_csv, low_memory=False)

        # Parse torsion IDs -> target_id + resid
        if "ID" in tor_df.columns and ("target_id" not in tor_df.columns or "resid" not in tor_df.columns):
            parsed = tor_df["ID"].apply(_parse_id_to_target_resid)
            tor_df["target_id"] = parsed.apply(lambda x: x[0])
            tor_df["resid"] = parsed.apply(lambda x: x[1])

        # Basic sanitation
        feat_df["target_id"] = feat_df["target_id"].astype(str).str.strip()
        feat_df["resid"] = pd.to_numeric(feat_df["resid"], errors="coerce").fillna(0).astype(int)
        tor_df["target_id"] = tor_df["target_id"].astype(str).str.strip()
        tor_df["resid"] = pd.to_numeric(tor_df["resid"], errors="coerce").fillna(0).astype(int)

        # Optionally keep only certain augment versions (clean/jittered/etc.)
        if self.use_versions is not None and "version" in feat_df.columns:
            feat_df["version"] = feat_df["version"].astype(str).str.strip()
            feat_df = feat_df[feat_df["version"].isin(self.use_versions)]

        # Keep only needed torsion columns
        for col in ["alpha", "beta"]:
            if col not in tor_df.columns:
                raise ValueError(f"torsion_csv must include column '{col}'")
            tor_df[col] = pd.to_numeric(tor_df[col], errors="coerce")

        # Clamp insane torsions and drop NaNs
        tor_df = tor_df.dropna(subset=["alpha", "beta"])
        tor_df = tor_df[(tor_df["alpha"].abs() <= 180) & (tor_df["beta"].abs() <= 180)]

        # Merge torsions onto features by (target_id, resid)
        merged = feat_df.merge(tor_df[["target_id", "resid", "alpha", "beta"]], on=["target_id", "resid"], how="inner")

        # Drop non-numeric / id columns from features
        drop_cols = set(ID_COLS + ["conformation"]) | self.feature_drop
        feature_cols = [c for c in merged.columns if c not in drop_cols and c not in ["alpha", "beta"]]
        numeric_cols = merged[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = numeric_cols

        # Group by target_id into variable-length sequences
        self.groups: Dict[str, pd.DataFrame] = {
            tid: g.sort_values("resid").reset_index(drop=True)
            for tid, g in merged.groupby("target_id")
        }
        self.keys = sorted(self.groups.keys())

        # Optional length filter
        if self.max_len is not None:
            self.keys = [k for k in self.keys if len(self.groups[k]) <= self.max_len]

        if len(self.keys) == 0:
            raise ValueError("No training samples after merging features with torsions. Check inputs.")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        tid = self.keys[idx]
        g = self.groups[tid]

        feats = g[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)  # [L,F]
        tors = g[["alpha", "beta"]].to_numpy(dtype=np.float32)               # [L,2]
        mask = np.ones((len(g),), dtype=np.bool_)                            # [L]

        return tid, torch.from_numpy(feats), torch.from_numpy(tors), torch.from_numpy(mask)


def pad_collate(batch):
    tids, feats, tors, masks = zip(*batch)
    max_len = max(x.shape[0] for x in feats)
    feat_dim = feats[0].shape[1]

    feats_pad = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    tors_pad = torch.zeros((len(batch), max_len, 2), dtype=torch.float32)
    mask_pad = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (f, t, m) in enumerate(zip(feats, tors, masks)):
        L = f.shape[0]
        feats_pad[i, :L] = f
        tors_pad[i, :L] = t
        mask_pad[i, :L] = m

    return list(tids), feats_pad, tors_pad, mask_pad


# -------------------------
# Training
# -------------------------

def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_edge_index(pos: torch.Tensor, mask: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Build a radius graph over flattened nodes.

    pos:  [B,L,3]
    mask: [B,L] bool
    returns edge_index: [2,E] over N = B*L nodes
    """
    B, L, _ = pos.shape
    pos_flat = pos.reshape(B * L, 3)

    # If we want to ignore padded nodes, we can set their coordinates far away,
    # which prevents most edges. (Simpler than subgraphing + remapping indices.)
    if mask is not None:
        mask_flat = mask.reshape(B * L)
        pos_flat = pos_flat.clone()
        pos_flat[~mask_flat] = 1e6  # push pads far away

    edge_index = radius_graph(pos_flat, r=radius, loop=False, max_num_neighbors=64)
    return edge_index


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seed_everything(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = RNATorsionFeatureDataset(
        feature_csv=args.features,
        torsion_csv=args.torsions,
        feature_drop=args.drop_cols,
        max_len=args.max_len,
        use_versions=args.use_versions,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 4),
        collate_fn=pad_collate,
        pin_memory=torch.cuda.is_available(),
    )

    egnn_cfg = EGNNConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    )
    model = EGNNTimeDenoiser(feature_dim=len(dataset.feature_cols), egnn_cfg=egnn_cfg, torsion_dim=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    schedule = make_linear_beta_schedule(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for _, feats, tors_true, mask in loader:
            feats = feats.to(device)
            tors_true = tors_true.to(device).clamp(-180.0, 180.0)
            mask = mask.to(device)

            B, L, _ = tors_true.shape

            # Random initial positions per batch (acts as a geometric scaffold for EGNN)
            pos = torch.randn((B, L, 3), device=device)

            edge_index = build_edge_index(pos, mask=mask, radius=args.radius).to(device)

            # Sample per-sample timestep t
            t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)

            # Forward noising
            x_t, _ = q_sample(tors_true, t, schedule)  # [B,L,2]

            # Predict x0 from x_t
            pred_x0 = model(feats=feats, pos=pos, edge_index=edge_index, x_t=x_t, t=t, T=args.T, mask=mask)

            # Loss: angle periodicity + smoothness
            loss_main = cyclic_mse_degrees(pred_x0[mask], tors_true[mask])
            loss_smooth = smoothness_l2(pred_x0, mask=mask)
            loss = loss_main + args.smooth_w * loss_smooth

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

        avg = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch:03d} | loss={avg:.6f}")

        # Save periodic checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = outdir / f"diffusion_denoiser_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save final
    final_path = outdir / "diffusion_denoiser.pt"
    torch.save(model.state_dict(), final_path)

    # Save config
    cfg = {
        "features": args.features,
        "torsions": args.torsions,
        "feature_cols": dataset.feature_cols,
        "egnn_cfg": asdict(egnn_cfg),
        "train_args": vars(args),
    }
    with open(outdir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"✅ Saved model to: {final_path}")
    print(f"✅ Saved config to: {outdir / 'train_config.json'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, required=True, help="Path to per-residue feature CSV.")
    p.add_argument("--torsions", type=str, required=True, help="Path to torsion label CSV.")
    p.add_argument("--outdir", type=str, default="checkpoints", help="Output directory for checkpoints.")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    # Model
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-layer-norm", action="store_true", help="Disable layer norm in EGNN layers.")

    # Diffusion schedule
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)

    # Graph building
    p.add_argument("--radius", type=float, default=12.0, help="Radius for radius_graph edges (Å units if coords are Å).")

    # Data filtering
    p.add_argument("--max-len", type=int, default=None, help="Drop sequences longer than this (optional).")
    p.add_argument("--use-versions", type=str, nargs="*", default=None,
                   help="If feature CSV includes a 'version' column, keep only these values (e.g., clean jittered).")
    p.add_argument("--drop-cols", type=str, nargs="*", default=[],
                   help="Extra columns to drop from feature inputs if present.")

    # Regularization
    p.add_argument("--smooth-w", type=float, default=0.01, help="Weight for smoothness regularization.")
    p.add_argument("--save-every", type=int, default=0, help="Save epoch checkpoints every N epochs (0 disables).")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
'''

out_path = "/mnt/data/train_diffusion.py"
Path = pathlib.Path
Path(out_path).write_text(code, encoding="utf-8")
out_path

