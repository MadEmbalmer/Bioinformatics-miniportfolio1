import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIG === #
COORD_DIR = "coords"
META_FILE = os.path.join(COORD_DIR, "coord_metadata.csv")
SAVE_PER_RESIDUE = "features/phase2_3/structure_features.csv"
SAVE_PER_CONFORMATION = "features/phase2_3/structure_features_summary.csv"
ERROR_LOG = "features/phase2_3/structure_feature_errors.log"
os.makedirs(os.path.dirname(SAVE_PER_RESIDUE), exist_ok=True)

CLASH_THRESHOLD = 1.5  # Ångstroms
ANGLE_MIN_VALID = 3
TORSION_MIN_VALID = 4

# === Helper Functions === #
def compute_bond_angles(coords):
    angles = [np.nan] * len(coords)
    for i in range(1, len(coords) - 1):
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        angles[i] = np.degrees(angle)
    return angles

def compute_dihedral_angles(coords):
    torsions = [np.nan] * len(coords)
    for i in range(len(coords) - 3):
        p0, p1, p2, p3 = coords[i:i+4]
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        n1 = np.cross(b0, b1)
        n2 = np.cross(b1, b2)
        if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0:
            continue
        m1 = np.cross(n1, b1 / np.linalg.norm(b1))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        torsions[i + 1] = np.degrees(np.arctan2(y, x))
    return torsions

def detect_clashes(coords, threshold=CLASH_THRESHOLD):
    clash_flags = [0] * len(coords)
    for i in range(len(coords)):
        for j in range(i + 2, len(coords)):  # skip immediate neighbors
            if np.linalg.norm(coords[i] - coords[j]) < threshold:
                clash_flags[i] += 1
                clash_flags[j] += 1
    return clash_flags

# === Load Metadata === #
print(" Loading coordinate metadata...")
meta = pd.read_csv(META_FILE)

records = []
summary_records = []
errors = []

print(" Extracting per-residue and per-conformation structural features...")
for _, row in tqdm(meta.iterrows(), total=len(meta)):
    tid = row["target_id"]
    conf_id = row["conformation"]
    coord_path = os.path.join(COORD_DIR, f"{tid}_conf{conf_id}_clean.npy")

    try:
        coords = np.load(coord_path)
        n = len(coords)

        bond_angles = compute_bond_angles(coords) if n >= ANGLE_MIN_VALID else [np.nan] * n
        torsions = compute_dihedral_angles(coords) if n >= TORSION_MIN_VALID else [np.nan] * n
        clashes = detect_clashes(coords)

        # Per-residue records
        for i in range(n):
            record = {
                "target_id": tid,
                "conformation": conf_id,
                "resid": i + 1,
                "bond_angle": bond_angles[i],
                "torsion_angle": torsions[i],
                "clash_score": clashes[i]
            }
            records.append(record)

        # Per-conformation summary
        bond_clean = [x for x in bond_angles if not np.isnan(x)]
        torsion_clean = [x for x in torsions if not np.isnan(x)]
        summary_records.append({
            "target_id": tid,
            "conformation": conf_id,
            "num_residues": n,
            "avg_bond_angle": np.mean(bond_clean) if bond_clean else np.nan,
            "std_bond_angle": np.std(bond_clean) if bond_clean else np.nan,
            "avg_torsion_angle": np.mean(torsion_clean) if torsion_clean else np.nan,
            "std_torsion_angle": np.std(torsion_clean) if torsion_clean else np.nan,
            "total_steric_clashes": np.sum(clashes)
        })

    except Exception as e:
        errors.append(f"{tid}_conf{conf_id}: {e}")
        print(f" Skipping {tid} conf {conf_id}: {e}")

# === Save Outputs === #
df_per_residue = pd.DataFrame(records)
df_per_residue.to_csv(SAVE_PER_RESIDUE, index=False)
print(f"\n Per-residue structure features saved to `{SAVE_PER_RESIDUE}` — {len(df_per_residue)} rows.")

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(SAVE_PER_CONFORMATION, index=False)
print(f" Per-conformation summaries saved to `{SAVE_PER_CONFORMATION}` — {len(df_summary)} entries.")

# === Save Errors === #
if errors:
    with open(ERROR_LOG, "w") as f:
        f.write("\n".join(errors))
    print(f" Logged {len(errors)} skipped conformations to `{ERROR_LOG}`.")
else:
    print(" All conformations processed without error.")
