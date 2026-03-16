import os
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
from Bio import pairwise2

# === CONFIG === #
BASE_DIR = "./output"
PLACEHOLDER_VALUE = -1.0e+18
ALLOWED_BASES = set("ACGU")
MERGE_IDENTITY_THRESHOLD = 0.98

# === Load Phase 0 outputs === #
def load_phase0_outputs():
    print("\U0001F4C2 Loading Phase 0 filtered files...")
    sequences = pd.read_csv(os.path.join(BASE_DIR, "filtered_train_sequences.csv"))
    labels = pd.read_csv(os.path.join(BASE_DIR, "filtered_train_labels.csv"))
    dup_map = pd.read_csv(os.path.join(BASE_DIR, "duplicate_sequence_groups.csv"))
    return sequences, labels, dup_map

# === Clean up invalid bases in sequences === #
def clean_sequences(df):
    print("\U0001F9EC Cleaning RNA sequences...")
    def clean(seq):
        return ''.join([b if b in ALLOWED_BASES else '' for b in seq])
    df["clean_sequence"] = df["sequence"].apply(clean)
    return df

# === Sequence identity function === #
def sequence_identity(seq1, seq2):
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    score = alignment[0].score
    return score / max(len(seq1), len(seq2))

# === RMSD via Kabsch algorithm === #
def compute_rmsd(P, Q):
    if P.shape != Q.shape or len(P) < 3:
        return np.inf
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    C = np.dot(P_centered.T, Q_centered)
    try:
        V, _, W = np.linalg.svd(C)
        if np.linalg.det(V) * np.linalg.det(W) < 0:
            V[:, -1] *= -1
        U = np.dot(V, W)
    except np.linalg.LinAlgError:
        return np.inf
    aligned = np.dot(P_centered, U)
    return np.sqrt(np.mean(np.sum((aligned - Q_centered) ** 2, axis=1)))

# === Merge map expansion (loose merge mode) === #
def filter_similar_sequences(dup_map, sequences, mode):
    print(f"\U0001F52A Filtering duplicates using merge mode: {mode}")
    if mode == "strict":
        return dup_map
    seq_dict = sequences.set_index("target_id")["clean_sequence"].to_dict()
    new_groups = defaultdict(list)
    visited = set()
    for base_id, seq1 in seq_dict.items():
        if base_id in visited:
            continue
        group = [base_id]
        for comp_id, seq2 in seq_dict.items():
            if comp_id == base_id or comp_id in visited:
                continue
            if sequence_identity(seq1, seq2) >= MERGE_IDENTITY_THRESHOLD:
                group.append(comp_id)
                visited.add(comp_id)
        if len(group) > 1:
            new_groups[seq1] += group
    dup_map = pd.DataFrame([(k, tid) for k, ids in new_groups.items() for tid in ids], columns=["sequence", "target_id"])
    return dup_map

# === Helper to extract conformer coordinates === #
def get_coords(block, conf_id="1"):
    return block[[f"{axis}_{conf_id}" for axis in ("x", "y", "z")]].values

# === Main align + merge === #
def align_and_merge_labels(seqs, labels, dup_map, rmsd_threshold):
    print("\U0001F9EC Aligning and merging coordinates across conformers...")

    labels[['target_id', 'resid']] = labels['ID'].str.extract(r'(.*)_(\d+)', expand=True)
    labels['resid'] = labels['resid'].astype(int)

    coord_sets = sorted(set(col.split("_")[1] for col in labels.columns if col.startswith("x_")))
    print(f"\U0001F50D Detected conformations: {coord_sets}")

    placeholder_mask = labels[[f"x_{conf}" for conf in coord_sets]].eq(PLACEHOLDER_VALUE).any(axis=1)
    labels = labels[~placeholder_mask]

    merged_labels = []
    merged_seqs = []
    seen_sequences = set()

    for sequence, group in dup_map.groupby("sequence"):
        target_ids = group["target_id"].tolist()
        base_id = target_ids[0]
        seq_entry = seqs[seqs["target_id"] == base_id].copy()

        label_blocks = []
        coords_sets = []

        for tid in target_ids:
            block = labels[labels["target_id"] == tid].sort_values("resid").reset_index(drop=True)
            if len(block) == 0:
                continue
            coords = get_coords(block, conf_id="1")
            coords_sets.append((tid, coords))
            label_blocks.append(block)

        if not label_blocks:
            continue

        ref_coords = coords_sets[0][1]
        structurally_valid_blocks = [label_blocks[0]]

        for tid, coords in coords_sets[1:]:
            if coords.shape != ref_coords.shape:
                continue
            rmsd = compute_rmsd(ref_coords, coords)
            if rmsd <= rmsd_threshold:
                block = labels[labels["target_id"] == tid].sort_values("resid").reset_index(drop=True)
                structurally_valid_blocks.append(block)

        if len(structurally_valid_blocks) < 2:
            continue

        merged = structurally_valid_blocks[0][["target_id", "resname", "resid"]].copy()
        for i, block in enumerate(structurally_valid_blocks):
            for axis in ['x', 'y', 'z']:
                merged[f"{axis}_{i+1}"] = block[f"{axis}_1"].values
        merged["target_id"] = base_id
        merged_labels.append(merged)
        seq_entry["target_id"] = base_id
        merged_seqs.append(seq_entry)
        seen_sequences.add(sequence)

    unique_seqs = seqs[~seqs["sequence"].isin(seen_sequences)].copy()
    unique_labels = labels[labels["target_id"].isin(unique_seqs["target_id"])]

    final_seqs = pd.concat(merged_seqs + [unique_seqs], ignore_index=True)
    final_labels = pd.concat(merged_labels + [unique_labels], ignore_index=True)

    # === FIX: trim label rows to match cleaned sequence lengths === #
    trimmed_labels = []
    seq_lens = dict(zip(final_seqs["target_id"], final_seqs["clean_sequence"].str.len()))
    for tid, group in final_labels.groupby("target_id"):
        if tid not in seq_lens:
            continue
        trimmed = group.sort_values("resid").head(seq_lens[tid])
        trimmed_labels.append(trimmed)
    final_labels = pd.concat(trimmed_labels, ignore_index=True)

    return final_seqs, final_labels

# === Entry point === #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_mode", choices=["strict", "loose"], default="strict",
                        help="Merge exact matches or allow near-duplicates")
    parser.add_argument("--rmsd_threshold", type=float, default=2.5,
                        help="Max RMSD allowed when merging structures")
    args = parser.parse_args()

    seq_df, lbl_df, dup_map = load_phase0_outputs()
    clean_seq_df = clean_sequences(seq_df)
    dup_map = filter_similar_sequences(dup_map, clean_seq_df, args.merge_mode)
    aligned_seqs, merged_labels = align_and_merge_labels(clean_seq_df, lbl_df, dup_map, args.rmsd_threshold)

    aligned_seqs.to_csv("phase1_cleaned_sequences.csv", index=False)
    merged_labels.to_csv("phase1_cleaned_labels.csv", index=False)

    print("\u2705 Phase 1.1 complete — merged and cleaned multi-conformation data saved.")
