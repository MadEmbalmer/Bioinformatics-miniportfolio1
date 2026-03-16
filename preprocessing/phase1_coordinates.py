import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# === CONFIG === #
SEQ_FILE = "phase1_cleaned_sequences.csv"
LABEL_FILE = "phase1_cleaned_labels.csv"
SAVE_COORDS = "coords"
JITTER_STDDEV = 0.2
ADVERSARIAL_STDDEV = 3.0
RMSD_ALIGN_THRESHOLD = 3.0
CLASH_DISTANCE = 1.5
CLASH_MAX_COUNT = 10
RMSD_MAX_ALLOWED = 8.0

os.makedirs(SAVE_COORDS, exist_ok=True)

print("📥 Loading Phase 1.1 cleaned sequences and labels...")
sequences = pd.read_csv(SEQ_FILE)
labels = pd.read_csv(LABEL_FILE)

coord_sets = sorted(set(col.split('_')[1] for col in labels.columns if col.startswith('x_')))
print(f"🔍 Detected {len(coord_sets)} conformations per structure: {coord_sets}")

# === Alignment & Noise Functions === #
def rmsd_center(coords):
    center = coords.mean(axis=0)
    return coords - center

def kabsch_align(P, Q):
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    C = np.dot(P_centered.T, Q_centered)
    try:
        V, S, W = np.linalg.svd(C)
        if np.linalg.det(V) * np.linalg.det(W) < 0:
            V[:, -1] *= -1
        U = np.dot(V, W)
    except np.linalg.LinAlgError:
        return Q
    return np.dot(Q_centered, U)

def apply_jitter(coords, std=JITTER_STDDEV):
    return coords + np.random.normal(0, std, coords.shape)

def apply_adversarial_noise(coords, std=ADVERSARIAL_STDDEV):
    return coords + np.random.normal(0, std, coords.shape)

def apply_random_rotation(coords):
    rot = R.random()
    return rot.apply(coords)

def compute_rmsd(A, B):
    return np.sqrt(((A - B) ** 2).sum(axis=1).mean())

def compute_clash_score(coords, threshold=CLASH_DISTANCE):
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    mask = np.triu(dists < threshold, k=2)
    count = np.sum(mask)
    return min(count, CLASH_MAX_COUNT + 1)

# === Main Processing Loop === #
print("\n🌐 Normalizing, aligning, and saving conformations...")
metadata = []

seq_dict = sequences.set_index("target_id")["clean_sequence" if "clean_sequence" in sequences.columns else "sequence"].to_dict()
label_groups = labels.groupby("target_id")
target_ids = sequences["target_id"].unique()

for tid in tqdm(target_ids, desc="Processing targets"):
    clean_seq = seq_dict[tid]
    seq_len = len(clean_seq)

    residues = label_groups.get_group(tid).sort_values("resid")

    coords_dict = {}
    for conf_id in coord_sets:
        try:
            cols = [f"x_{conf_id}", f"y_{conf_id}", f"z_{conf_id}"]
            coords = residues[cols].to_numpy().astype(np.float32)
            if coords.shape[0] != seq_len:
                print(f"⚠️ Skipping {tid} conf {conf_id}: coord mismatch (seq_len={seq_len}, coords={coords.shape[0]})")
                continue
            coords_dict[conf_id] = coords
        except Exception as e:
            print(f"❌ Error reading {tid} conf {conf_id}: {e}")

    if not coords_dict:
        continue

    ref_conf = sorted(coords_dict)[0]
    ref_coords = rmsd_center(coords_dict[ref_conf])

    for conf_id, coords in coords_dict.items():
        try:
            centered = rmsd_center(coords)
            aligned = kabsch_align(ref_coords, centered)

            clean_file = f"{tid}_conf{conf_id}_clean.npy"
            np.save(os.path.join(SAVE_COORDS, clean_file), aligned)
            metadata.append({
                "target_id": tid,
                "conformation": conf_id,
                "version": "clean",
                "num_residues": seq_len,
                "is_valid": 1,
                "file": clean_file,
                "rmsd": 0.0,
                "clashes": 0
            })

            jittered = apply_jitter(aligned)
            jittered_rmsd = compute_rmsd(aligned, jittered)
            jittered_clash = compute_clash_score(jittered)
            jittered_valid = int(jittered_rmsd <= RMSD_MAX_ALLOWED and jittered_clash <= CLASH_MAX_COUNT)
            jittered_file = f"{tid}_conf{conf_id}_jittered.npy"
            if jittered_valid:
                np.save(os.path.join(SAVE_COORDS, jittered_file), jittered)
            metadata.append({
                "target_id": tid,
                "conformation": conf_id,
                "version": "jittered",
                "num_residues": seq_len,
                "is_valid": jittered_valid,
                "file": jittered_file,
                "rmsd": jittered_rmsd,
                "clashes": jittered_clash
            })

            adversarial = apply_adversarial_noise(aligned)
            adv_rmsd = compute_rmsd(aligned, adversarial)
            adv_clash = compute_clash_score(adversarial)
            adv_file = f"{tid}_conf{conf_id}_adversarial.npy"
            np.save(os.path.join(SAVE_COORDS, adv_file), adversarial)
            metadata.append({
                "target_id": tid,
                "conformation": conf_id,
                "version": "adversarial",
                "num_residues": seq_len,
                "is_valid": 0,
                "file": adv_file,
                "rmsd": adv_rmsd,
                "clashes": adv_clash
            })

            rotated = apply_random_rotation(aligned)
            rot_rmsd = compute_rmsd(aligned, rotated)
            rot_clash = compute_clash_score(rotated)
            rot_valid = int(rot_rmsd <= RMSD_MAX_ALLOWED and rot_clash <= CLASH_MAX_COUNT)
            rot_file = f"{tid}_conf{conf_id}_rotated.npy"
            if rot_valid:
                np.save(os.path.join(SAVE_COORDS, rot_file), rotated)
            metadata.append({
                "target_id": tid,
                "conformation": conf_id,
                "version": "rotated",
                "num_residues": seq_len,
                "is_valid": rot_valid,
                "file": rot_file,
                "rmsd": rot_rmsd,
                "clashes": rot_clash
            })

        except Exception as e:
            print(f"❌ Failed to align or save {tid} conf {conf_id}: {e}")

# === Save Metadata === #
meta_df = pd.DataFrame(metadata)
meta_df.to_csv(os.path.join(SAVE_COORDS, "coord_metadata.csv"), index=False)

if not meta_df.empty and "target_id" in meta_df.columns:
    print(f"\n✅ Saved coordinates and metadata for {len(metadata)} conformations across {meta_df['target_id'].nunique()} sequences.")
else:
    print("\n⚠️ No metadata saved — possibly all coordinate reads failed.")
