import os
import pandas as pd
import numpy as np
from datetime import datetime

# === CONFIG === #
BASE_DIR = "data" # place Kaggle competition CSV files here
SAVE_DIR = "./output"
os.makedirs(SAVE_DIR, exist_ok=True)

TEMPORAL_CUTOFF_SAFE_DATE = pd.to_datetime("2022-05-27")
PLACEHOLDER_VALUE = -1.0e+18
NOISE_DISTANCE_THRESHOLD = 30.0 # Angstroms
RMSD_THRESHOLD = 2.5 # Optional threshold for near-duplicate removal

# === Load Data === #
def load_data():
 print("\U0001F4C5 Loading all sequence and label files...")
 
 # Load and tag v1
 train_sequences_v1 = pd.read_csv(os.path.join(BASE_DIR, "train_sequences.csv"))
 train_sequences_v1["source"] = "v1"
 train_labels_v1 = pd.read_csv(os.path.join(BASE_DIR, "train_labels.csv"))
 train_labels_v1["source"] = "v1"

 # Load and tag v2
 train_sequences_v2 = pd.read_csv(os.path.join(BASE_DIR, "train_sequences.v2.csv"))
 train_sequences_v2["source"] = "v2"
 train_labels_v2 = pd.read_csv(os.path.join(BASE_DIR, "train_labels.v2.csv"))
 train_labels_v2["source"] = "v2"

 # Combine
 train_sequences = pd.concat([train_sequences_v1, train_sequences_v2], ignore_index=True)
 train_labels = pd.concat([train_labels_v1, train_labels_v2], ignore_index=True)
 train_labels["target_id"] = train_labels["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))

 val_sequences = pd.read_csv(os.path.join(BASE_DIR, "validation_sequences.csv"))
 val_labels = pd.read_csv(os.path.join(BASE_DIR, "validation_labels.csv"))

 test_sequences = pd.read_csv(os.path.join(BASE_DIR, "test_sequences.csv"))
 sample_submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

 print(" Loaded and stacked v1 + v2 files successfully.")
 return train_sequences, train_labels, val_sequences, val_labels, test_sequences, sample_submission

# === Filter by temporal cutoff === #
def clean_temporal(train_sequences):
 print(" Filtering sequences by temporal cutoff...")
 train_sequences["temporal_cutoff"] = pd.to_datetime(train_sequences["temporal_cutoff"], errors='coerce')
 filtered = train_sequences[train_sequences["temporal_cutoff"] <= TEMPORAL_CUTOFF_SAFE_DATE].copy()
 print(f" Kept {len(filtered)} / {len(train_sequences)} sequences before cutoff.")
 return filtered

# === Filter invalid rows with any placeholder values in any conformation === #
def filter_valid_labels(labels):
 print(" Filtering invalid placeholder coordinates...")
 coord_cols = [col for col in labels.columns if col.startswith("x_") or col.startswith("y_") or col.startswith("z_")]
 valid_mask = ~(labels[coord_cols] == PLACEHOLDER_VALUE).any(axis=1)
 valid_labels = labels[valid_mask].copy()
 print(f" Retained {len(valid_labels)} valid coordinate rows out of {len(labels)}.")
 return valid_labels

# === Remove noisy structures by C1' distance jumps in x_1 conformation === #
def remove_noisy_structures(filtered_sequences, valid_labels):
 print(" Removing noisy/broken structures...")
 clean_target_ids = []

 for tid in filtered_sequences["target_id"].values:
 coords = valid_labels[valid_labels["target_id"] == tid][['x_1', 'y_1', 'z_1']].values
 if len(coords) < 2:
 continue
 dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
 if (dists > NOISE_DISTANCE_THRESHOLD).sum() / len(dists) > 0.05:
 continue
 clean_target_ids.append(tid)

 filtered_sequences = filtered_sequences[filtered_sequences["target_id"].isin(clean_target_ids)].copy()
 valid_labels = valid_labels[valid_labels["target_id"].isin(clean_target_ids)].copy()
 print(f" Retained {len(clean_target_ids)} targets after noise filtering.")
 return filtered_sequences, valid_labels

# === Require full structure coverage in x_1 conformation === #
def keep_sequences_with_full_coords(filtered_sequences, valid_labels):
 print(" Matching sequences with complete coordinate sets...")
 seq_lengths = filtered_sequences.set_index("target_id")["sequence"].apply(len).to_dict()
 coord_counts = valid_labels.groupby("target_id").size().to_dict()

 valid_ids = [
 tid for tid in seq_lengths
 if tid in coord_counts and coord_counts[tid] == seq_lengths[tid]
 ]

 filtered_sequences = filtered_sequences[filtered_sequences["target_id"].isin(valid_ids)].copy()
 valid_labels = valid_labels[valid_labels["target_id"].isin(valid_ids)].copy()
 print(f" Retained {len(valid_ids)} fully matched targets with complete structures.")
 return filtered_sequences, valid_labels

# === Add basic features === #
def add_sequence_length_features(sequences):
 print(" Adding sequence length features...")
 sequences["seq_length"] = sequences["sequence"].apply(len)
 sequences["log_seq_length"] = np.log(sequences["seq_length"] + 1)
 return sequences

# === Optional: Remove structural near-duplicates via centroid RMSD === #
def deduplicate_structural_near_duplicates(sequences, labels, threshold=RMSD_THRESHOLD):
 print(" Checking for structurally redundant targets...")
 from scipy.spatial.distance import pdist, squareform

 coords_dict = {}
 for tid in sequences["target_id"].values:
 coords = labels[labels["target_id"] == tid][['x_1', 'y_1', 'z_1']].values
 if len(coords) == 0:
 continue
 coords_dict[tid] = coords

 unique_tids = []
 seen = set()

 tids = list(coords_dict.keys())
 for i, tid1 in enumerate(tids):
 if tid1 in seen:
 continue
 unique_tids.append(tid1)
 coords1 = coords_dict[tid1]
 for j in range(i + 1, len(tids)):
 tid2 = tids[j]
 if tid2 in seen or coords1.shape != coords_dict[tid2].shape:
 continue
 coords2 = coords_dict[tid2]
 rmsd = np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))
 if rmsd < threshold:
 seen.add(tid2)

 deduped_sequences = sequences[sequences["target_id"].isin(unique_tids)].copy()
 deduped_labels = labels[labels["target_id"].isin(unique_tids)].copy()
 print(f" Removed {len(tids) - len(unique_tids)} near-duplicate structures.")
 return deduped_sequences, deduped_labels

# === Entry Point === #
if __name__ == "__main__":
 train_seq, train_lbls, val_seq, val_lbls, test_seq, sample_sub = load_data()

 filtered_train_seq = clean_temporal(train_seq)
 valid_train_lbls = filter_valid_labels(train_lbls)
 filtered_train_seq, valid_train_lbls = remove_noisy_structures(filtered_train_seq, valid_train_lbls)
 filtered_train_seq, valid_train_lbls = keep_sequences_with_full_coords(filtered_train_seq, valid_train_lbls)
 filtered_train_seq = add_sequence_length_features(filtered_train_seq)

 print(" Checking for duplicate sequences across different target_ids...")
 dup_groups = filtered_train_seq.groupby("sequence")["target_id"].apply(list)
 dup_groups = dup_groups[dup_groups.apply(len) > 1]

 if not dup_groups.empty:
 print(f" Found {len(dup_groups)} duplicate sequence groups.")
 dup_df = dup_groups.reset_index().explode("target_id")
 dup_df.to_csv(os.path.join(SAVE_DIR, "duplicate_sequence_groups.csv"), index=False)
 else:
 print(" No duplicate sequences found.")

 # Remove near-duplicate structures
 filtered_train_seq, valid_train_lbls = deduplicate_structural_near_duplicates(filtered_train_seq, valid_train_lbls)

 # === Save Outputs === #
 filtered_train_seq.to_csv(os.path.join(SAVE_DIR, "filtered_train_sequences.csv"), index=False)
 valid_train_lbls.to_csv(os.path.join(SAVE_DIR, "filtered_train_labels.csv"), index=False)
 val_seq.to_csv(os.path.join(SAVE_DIR, "validation_sequences.csv"), index=False)
 val_lbls.to_csv(os.path.join(SAVE_DIR, "validation_labels.csv"), index=False)

 print(" Phase 0 complete — upgraded clean sequences and labels saved.")
