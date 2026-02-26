import pathlib, textwrap

code = r'''#!/usr/bin/env python3
"""
Sample RNA backbone structures (C1' trace) using a trained diffusion torsion model.

This is a clean, portfolio-friendly inference entry point.

Pipeline:
  features (per-residue) -> EGNN diffusion denoiser -> sampled torsions (alpha/beta)
  -> internal-coordinate decoder -> 3D coordinates (C1' atoms)

Inputs (not bundled):
- Feature CSV, e.g. features/final_merged_features.csv
- Trained checkpoint, e.g. checkpoints/diffusion_denoiser.pt

Outputs:
- PDB files per target/sample (optional)
- Kaggle-style submission.csv (optional)

Example (PDB export):
    python inference/sample_structures.py \
      --features features/final_merged_features.csv \
      --checkpoint checkpoints/diffusion_denoiser.pt \
      --outdir outputs \
      --num-samples 20 --num-final 5

Example (submission.csv):
    python inference/sample_structures.py \
      --features features/final_merged_features.csv \
      --checkpoint checkpoints/diffusion_denoiser.pt \
      --outdir outputs \
      --make-submission

Notes
-----
- This script is data-agnostic: it does not download Kaggle data.
- It assumes 'resname' exists in the feature CSV (A/C/G/U), and uses it in outputs.
- Graph edges are built from random initial positions using a radius graph.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.nn import radius_graph
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for radius_graph. "
        "Install with: pip install torch-geometric"
    ) from e

from sklearn.cluster import KMeans

# Make project root importable so `models.*` and `utils.*` work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion_model import (  # noqa: E402
    EGNNTimeDenoiser,
    make_linear_beta_schedule,
    sample_torsions,
)
from models.egnn import EGNNConfig  # noqa: E402
from utils.internal_coordinate_decoder import torsion_to_coords  # noqa: E402


BASE_MAP = {"A": "A", "C": "C", "G": "G", "U": "U"}


def write_pdb_c1prime(coords: np.ndarray, seq: List[str], path: Path) -> None:
    """
    Write a minimal PDB containing only C1' atoms for an RNA chain.
    coords: [L,3]
    seq: list of 'A','C','G','U'
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (xyz, base) in enumerate(zip(coords, seq), start=1):
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            res = BASE_MAP.get(str(base).upper(), "N")
            # Simple PDB line (ATOM records). Chain 'A' by default.
            f.write(
                f"ATOM  {i:5d}  C1'  {res:>1s} A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


def load_targets(feature_csv: str, use_conformation: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load features and group by target_id.
    Optionally filter a specific conformation if column exists.
    """
    df = pd.read_csv(feature_csv, low_memory=False)
    df["target_id"] = df["target_id"].astype(str).str.strip()
    if "resid" in df.columns:
        df["resid"] = pd.to_numeric(df["resid"], errors="coerce").fillna(0).astype(int)

    if use_conformation is not None and "conformation" in df.columns:
        df["conformation"] = df["conformation"].astype(str).str.strip()
        df = df[df["conformation"] == str(use_conformation)]

    # Ensure sorting for consistent residue order
    groups = {
        tid: g.sort_values("resid").reset_index(drop=True)
        for tid, g in df.groupby("target_id")
    }
    return groups


def get_feature_matrix(group: pd.DataFrame, drop_cols: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Extract numeric feature matrix [L,F], plus sequence bases and residue ids.
    """
    seq = group["resname"].astype(str).tolist() if "resname" in group.columns else ["N"] * len(group)
    resid = group["resid"].astype(int).tolist() if "resid" in group.columns else list(range(1, len(group) + 1))

    drop = set(["target_id", "resid", "resname", "conformation"] + list(drop_cols))
    feat_df = group.drop(columns=[c for c in drop if c in group.columns], errors="ignore")

    # Keep only numeric features
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = feat_df.to_numpy(dtype=np.float32)
    return X, seq, resid


def build_edge_index_from_pos(pos: torch.Tensor, radius: float) -> torch.Tensor:
    """
    pos: [L,3] -> edge_index [2,E]
    """
    return radius_graph(pos, r=radius, loop=False, max_num_neighbors=64)


def pick_representatives(coords_list: List[np.ndarray], k: int, seed: int = 42) -> List[int]:
    """
    Pick k representative samples via KMeans in flattened coordinate space.
    coords_list: list of [L,3]
    returns indices into coords_list
    """
    if len(coords_list) <= k:
        return list(range(len(coords_list)))

    X = np.stack([c.reshape(-1) for c in coords_list])  # [S, 3L]
    try:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(X)
        centers = km.cluster_centers_
        chosen = []
        for center in centers:
            idx = int(np.argmin(np.linalg.norm(X - center[None, :], axis=1)))
            chosen.append(idx)
        # Ensure uniqueness while preserving order
        seen = set()
        uniq = []
        for i in chosen:
            if i not in seen:
                uniq.append(i); seen.add(i)
        while len(uniq) < k:
            # fill with remaining closest-to-any-center
            for i in range(len(coords_list)):
                if i not in seen:
                    uniq.append(i); seen.add(i)
                if len(uniq) == k:
                    break
        return uniq[:k]
    except Exception:
        return list(range(k))


def to_submission_rows(
    target_id: str,
    seq: List[str],
    resid: List[int],
    coords_final: List[np.ndarray],
) -> List[dict]:
    """
    Create Kaggle submission rows for one target_id.
    coords_final: list length K, each [L,3]
    """
    K = len(coords_final)
    L = len(seq)
    rows = []
    for i in range(L):
        row = {
            "ID": f"{target_id}_{i+1}",
            "resname": str(seq[i]).upper(),
            "resid": int(resid[i]),
        }
        for k in range(K):
            row[f"x_{k+1}"] = float(coords_final[k][i, 0])
            row[f"y_{k+1}"] = float(coords_final[k][i, 1])
            row[f"z_{k+1}"] = float(coords_final[k][i, 2])
        rows.append(row)
    return rows


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load features
    targets = load_targets(args.features, use_conformation=args.use_conformation)
    if args.limit is not None:
        # take first N targets for quick tests
        keys = sorted(list(targets.keys()))[: int(args.limit)]
        targets = {k: targets[k] for k in keys}

    # Build model
    egnn_cfg = EGNNConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    )

    # Infer feature_dim from the first target
    first_tid = next(iter(targets.keys()))
    X0, _, _ = get_feature_matrix(targets[first_tid], drop_cols=args.drop_cols)
    feature_dim = X0.shape[1]

    model = EGNNTimeDenoiser(feature_dim=feature_dim, egnn_cfg=egnn_cfg, torsion_dim=2).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    schedule = make_linear_beta_schedule(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    submission_rows: List[dict] = []

    for tid, group in targets.items():
        X, seq, resid = get_feature_matrix(group, drop_cols=args.drop_cols)
        L = X.shape[0]

        feats = torch.from_numpy(X).to(device).unsqueeze(0)  # [1,L,F]
        mask = torch.ones((1, L), dtype=torch.bool, device=device)

        # Random initial positions for graph construction
        pos = torch.randn((1, L, 3), device=device)
        edge_index = build_edge_index_from_pos(pos.reshape(L, 3), radius=args.radius).to(device)

        # Sample S torsion configurations and decode to coords
        coords_samples: List[np.ndarray] = []
        for s in range(args.num_samples):
            tors = sample_torsions(
                model=model,
                feats=feats,
                pos=pos,
                edge_index=edge_index,
                schedule=schedule,
                T=args.T,
                steps=args.steps,
                mask=mask,
                clamp_degrees=args.clamp,
            )  # [1,L,2] degrees

            coords = torsion_to_coords(tors.squeeze(0), mask=None).detach().cpu().numpy()  # [L,3]
            coords_samples.append(coords)

        # Pick K representative conformations
        chosen_idx = pick_representatives(coords_samples, k=args.num_final, seed=args.seed)
        coords_final = [coords_samples[i] for i in chosen_idx]

        # Save PDBs if requested
        if args.write_pdb:
            tdir = outdir / "pdb" / tid
            for j, coords in enumerate(coords_final, start=1):
                write_pdb_c1prime(coords, seq, tdir / f"conf_{j}.pdb")

        # Append submission rows if requested
        if args.make_submission:
            submission_rows.extend(to_submission_rows(tid, seq, resid, coords_final))

        print(f"✅ {tid}: sampled {args.num_samples} -> selected {len(coords_final)}")

    if args.make_submission:
        sub_path = outdir / "submission.csv"
        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(sub_path, index=False)
        print(f"✅ Wrote submission to: {sub_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, required=True, help="Per-residue feature CSV.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained diffusion_denoiser.pt.")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory.")

    p.add_argument("--num-samples", type=int, default=20, help="How many samples to draw per target.")
    p.add_argument("--num-final", type=int, default=5, help="How many representative conformations to keep.")
    p.add_argument("--steps", type=int, default=200, help="Reverse diffusion steps (subset of timesteps).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU.")

    # Diffusion schedule
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--clamp", type=float, default=180.0, help="Clamp torsions to [-clamp, clamp] degrees.")

    # Graph building
    p.add_argument("--radius", type=float, default=12.0, help="Radius for radius_graph.")

    # Model
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-layer-norm", action="store_true")

    # CSV handling
    p.add_argument("--use-conformation", type=str, default=None,
                   help="If the feature CSV has 'conformation', keep only this value (e.g. '0' or '1').")
    p.add_argument("--drop-cols", type=str, nargs="*", default=[],
                   help="Extra columns to drop from input feature matrix if present.")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N targets (quick test).")

    # Outputs
    p.add_argument("--write-pdb", action="store_true", help="Write PDBs for selected conformations.")
    p.add_argument("--make-submission", action="store_true", help="Write Kaggle-style submission.csv.")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_inference(args)
'''
out_path = "/mnt/data/sample_structures.py"
pathlib.Path(out_path).write_text(code, encoding="utf-8")
out_path

