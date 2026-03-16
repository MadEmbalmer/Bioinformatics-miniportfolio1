import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# === CONFIG === #
COORD_DIR = "coords"
META_FILE = os.path.join(COORD_DIR, "coord_metadata.csv")
SAVE_PER_RESIDUE = "features/phase2_3/structure_features.csv"
SAVE_PER_CONFORMATION = "features/phase2_3/structure_features_summary.csv"
ERROR_LOG = "features/phase2_3/structure_feature_errors.log"
os.makedirs(os.path.dirname(SAVE_PER_RESIDUE), exist_ok=True)

CLASH_THRESHOLD = 1.5
CONTACT_THRESHOLD = 8.0
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
        for j in range(i + 2, len(coords)):
            if np.linalg.norm(coords[i] - coords[j]) < threshold:
                clash_flags[i] += 1
                clash_flags[j] += 1
    return clash_flags

def compute_contact_map(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    contact_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
            if dist <= CONTACT_THRESHOLD:
                contact_matrix[i, j] = contact_matrix[j, i] = 1
    return dist_matrix, contact_matrix

def compute_curvature(coords):
    curvatures = [np.nan] * len(coords)
    for i in range(1, len(coords) - 1):
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        ab = b - a
        bc = c - b
        ab_norm = np.linalg.norm(ab)
        bc_norm = np.linalg.norm(bc)
        if ab_norm == 0 or bc_norm == 0:
            continue
        cosine_angle = np.dot(ab, bc) / (ab_norm * bc_norm)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        if angle > 0:
            sin_half_angle = np.sin(angle / 2)
            if sin_half_angle != 0:
                radius = ab_norm / (2 * sin_half_angle)
                if radius != 0:
                    curvatures[i] = 1 / radius
    return curvatures

def process_row(row):
    try:
        tid = row["target_id"]
        conf_id = row["conformation"]
        coord_path = os.path.join(COORD_DIR, f"{tid}_conf{conf_id}_clean.npy")

        coords = np.load(coord_path)
        n = len(coords)

        bond_angles = compute_bond_angles(coords) if n >= ANGLE_MIN_VALID else [np.nan] * n
        torsions = compute_dihedral_angles(coords) if n >= TORSION_MIN_VALID else [np.nan] * n
        curvature = compute_curvature(coords) if n >= ANGLE_MIN_VALID else [np.nan] * n

        torsion_clean = [x for x in torsions if not np.isnan(x)]
        bond_clean = [x for x in bond_angles if not np.isnan(x)]
        curvature_clean = [x for x in curvature if not np.isnan(x)]

        mean_torsion_dev = np.std(torsion_clean) if torsion_clean else np.nan
        clashes = detect_clashes(coords)
        dist_matrix, contact_matrix = compute_contact_map(coords)
        total_contacts = np.sum(np.triu(contact_matrix, k=1))
        total_possible = n * (n - 1) // 2
        long_range_contacts = np.sum([
            contact_matrix[i, j]
            for i in range(n)
            for j in range(i + 8, n)
        ])

        per_residue = [{
            "target_id": tid,
            "conformation": conf_id,
            "resid": i + 1,
            "bond_angle": bond_angles[i],
            "torsion_angle": torsions[i],
            "curvature": curvature[i],
            "clash_score": clashes[i]
        } for i in range(n)]

        summary = {
            "target_id": tid,
            "conformation": conf_id,
            "num_residues": n,
            "avg_bond_angle": np.mean(bond_clean) if bond_clean else np.nan,
            "std_bond_angle": np.std(bond_clean) if bond_clean else np.nan,
            "avg_torsion_angle": np.mean(torsion_clean) if torsion_clean else np.nan,
            "std_torsion_angle": np.std(torsion_clean) if torsion_clean else np.nan,
            "mean_abs_torsion_deviation": mean_torsion_dev,
            "avg_curvature": np.nanmean(curvature_clean) if curvature_clean else np.nan,
            "std_curvature": np.nanstd(curvature_clean) if curvature_clean else np.nan,
            "total_steric_clashes": np.sum(clashes),
            "total_contacts": total_contacts,
            "long_range_contact_pct": long_range_contacts / total_contacts if total_contacts else 0.0,
            "contact_density": total_contacts / total_possible if total_possible else 0.0
        }

        return per_residue, summary, None

    except Exception as e:
        return [], None, f"{row['target_id']}_conf{row['conformation']}: {e}"

# === MAIN EXECUTION === #
def main():
    print("📦 Loading coordinate metadata...")
    meta = pd.read_csv(META_FILE)

    records = []
    summary_records = []
    errors = []

    print("🧜 Extracting per-residue and per-conformation structural features...")

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_row, row) for _, row in meta.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res, summary, err = future.result()
            if err:
                errors.append(err)
            else:
                records.extend(res)
                summary_records.append(summary)

    df_per_residue = pd.DataFrame(records)
    df_per_residue.to_csv(SAVE_PER_RESIDUE, index=False)
    print(f"\n✅ Per-residue structure features saved to `{SAVE_PER_RESIDUE}` — {len(df_per_residue)} rows.")

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(SAVE_PER_CONFORMATION, index=False)
    print(f"✅ Per-conformation summaries saved to `{SAVE_PER_CONFORMATION}` — {len(df_summary)} entries.")

    if errors:
        with open(ERROR_LOG, "w") as f:
            f.write("\n".join(errors))
        print(f"⚠️ Logged {len(errors)} skipped conformations to `{ERROR_LOG}`.")
    else:
        print("✅ All conformations processed without error.")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Only necessary for frozen apps but safe to include
    main()
