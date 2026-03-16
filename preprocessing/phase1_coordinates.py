import os
import pandas as pd
import numpy as np

# === CONFIG === #
SEQ_FILE = "phase1_cleaned_sequences.csv"
LABEL_FILE = "phase1_cleaned_labels.csv"
SAVE_COORDS = "coords"
JITTER_STDDEV = 0.2  # Angstroms of Gaussian noise

# === Ensure save dir exists === #
os.makedirs(SAVE_COORDS, exist_ok=True)

print(" Loading Phase 1.1 cleaned sequences and labels...")
sequences = pd.read_csv(SEQ_FILE)
labels = pd.read_csv(LABEL_FILE)

# === Extract conformation keys (e.g. x_1, x_2, ...) === #
coord_sets = sorted(set(col.split('_')[1] for col in labels.columns if col.startswith('x_')))
print(f" Detected {len(coord_sets)} conformations per structure: {coord_sets}")

# === NORMALIZATION FUNCTIONS === #
def rmsd_center(coords):
    center = coords.mean(axis=0)
    return coords - center

def apply_jitter(coords, std=JITTER_STDDEV):
    return coords + np.random.normal(0, std, coords.shape)

# === NORMALIZE AND SAVE PER-CONFORMATION COORDINATES === #
print("\n Normalizing and saving multi-conformation coordinates...")
metadata = []

for tid in sequences["target_id"].unique():
    if "clean_sequence" in sequences.columns:
        clean_seq = sequences.loc[sequences["target_id"] == tid, "clean_sequence"].values[0]
    else:
        clean_seq = sequences.loc[sequences["target_id"] == tid, "sequence"].values[0]
    seq_len = len(clean_seq)
    
    residues = labels[labels["target_id"] == tid].sort_values("resid")

    for conf_id in coord_sets:
        try:
            coords = residues[[f"x_{conf_id}", f"y_{conf_id}", f"z_{conf_id}"]].values.astype(np.float32)

            if coords.shape[0] != seq_len:
                print(f" Skipping {tid} conformation {conf_id}: coord mismatch (seq_len={seq_len}, coords={coords.shape[0]})")
                continue

            coords_clean = rmsd_center(coords)
            coords_jittered = apply_jitter(coords_clean)

            # Save both versions
            np.save(os.path.join(SAVE_COORDS, f"{tid}_conf{conf_id}_clean.npy"), coords_clean)
            np.save(os.path.join(SAVE_COORDS, f"{tid}_conf{conf_id}_jittered.npy"), coords_jittered)

            metadata.append({
                "target_id": tid,
                "conformation": conf_id,
                "num_residues": seq_len
            })
        except Exception as e:
            print(f" Error processing {tid} conformation {conf_id}: {e}")

# === Save Metadata === #
meta_df = pd.DataFrame(metadata)
meta_df.to_csv(os.path.join(SAVE_COORDS, "coord_metadata.csv"), index=False)

print(f"\n Saved coordinates for {len(metadata)} conformations across {meta_df['target_id'].nunique()} sequences.")
print(" Phase 1.3 complete — centered, jittered, and clean coords saved per conformation.")
