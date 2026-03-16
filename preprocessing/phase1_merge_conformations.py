import os
import pandas as pd
import numpy as np
from collections import defaultdict

# === CONFIG === #
BASE_DIR = "data"  # place Kaggle competition data here
PLACEHOLDER_VALUE = -1.0e+18
ALLOWED_BASES = set("ACGU")

# === LOAD DATA === #
def load_phase0_outputs():
    print(" Loading Phase 0 filtered files...")
    sequences = pd.read_csv(os.path.join(BASE_DIR, "filtered_train_sequences_v2.csv"))  # <-- changed
    labels = pd.read_csv(os.path.join(BASE_DIR, "filtered_train_labels_v2.csv"))          # <-- changed
    duplicate_map = pd.read_csv(os.path.join(BASE_DIR, "duplicate_sequence_groups.csv"))
    return sequences, labels, duplicate_map


# === CLEAN SEQUENCES === #
def clean_sequences(df):
    print("\U0001F9EC Cleaning RNA sequences...")
    def clean(seq):
        return ''.join([b if b in ALLOWED_BASES else '' for b in seq])
    df["clean_sequence"] = df["sequence"].apply(clean)
    return df

# === FIXED PARSING + LABEL HANDLING === #
def align_and_merge_labels(seqs, labels, dup_map):
    print(" Aligning and merging coordinates...")

    # Fix target_id and resid extraction
    labels[['target_id', 'resid']] = labels['ID'].str.extract(r'(.*)_(\d+)', expand=True)
    labels['resid'] = labels['resid'].astype(int)

    # Remove invalid coordinate rows
    coord_cols = [col for col in labels.columns if col.startswith('x_')]
    coord_base = [col.replace('x_', '') for col in coord_cols]
    placeholder_mask = labels[[f"x_{i}" for i in range(1, len(coord_cols) + 1)]].eq(PLACEHOLDER_VALUE).any(axis=1)
    labels = labels[~placeholder_mask]

    # Merge conformations for each sequence group
    merged_labels = []
    merged_seqs = []
    seen_seq = set()

    for sequence, group in dup_map.groupby("sequence"):
        target_ids = group["target_id"].tolist()
        base_target_id = target_ids[0]
        seq_entry = seqs[seqs["target_id"] == base_target_id].copy()

        # Gather labels across all target_ids for this sequence
        label_blocks = [labels[labels["target_id"] == tid].sort_values("resid").reset_index(drop=True)
                        for tid in target_ids if tid in labels["target_id"].values]

        # Merge x_1, x_2,... from different conformations
        if not label_blocks:
            continue

        merged = label_blocks[0][["target_id", "resname", "resid"]].copy()
        for i, block in enumerate(label_blocks):
            for axis in ['x', 'y', 'z']:
                merged[f"{axis}_{i+1}"] = block[f"{axis}_1"].values  # use _1 from source

        merged["target_id"] = base_target_id  # unify ID
        merged_labels.append(merged)
        seq_entry["target_id"] = base_target_id
        merged_seqs.append(seq_entry)
        seen_seq.add(sequence)

    # Include non-duplicate sequences
    unique_seqs = seqs[~seqs["sequence"].isin(seen_seq)].copy()
    unique_labels = labels[labels["target_id"].isin(unique_seqs["target_id"])]

    final_seqs = pd.concat([*merged_seqs, unique_seqs], ignore_index=True)
    final_labels = pd.concat([*merged_labels, unique_labels], ignore_index=True)

    return final_seqs, final_labels

# === RUN === #
if __name__ == "__main__":
    seq_df, lbl_df, dup_map = load_phase0_outputs()
    clean_seq_df = clean_sequences(seq_df)
    aligned_seqs, merged_labels = align_and_merge_labels(clean_seq_df, lbl_df, dup_map)

    aligned_seqs.to_csv("phase1_cleaned_sequences.csv", index=False)
    merged_labels.to_csv("phase1_cleaned_labels.csv", index=False)

    print("\n\u2705 Phase 1.1 complete — merged and cleaned multi-conformation data saved.")
