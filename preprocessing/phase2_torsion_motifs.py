import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === CONFIG === #
SEQ_FILE = "phase1_cleaned_sequences.csv"
META_FILE = "coords/coord_metadata.csv"
COORD_PATH = "coords"
RNA_FEAT_FILE = "features/phase2_2/rna_aware_features.csv"
OUT_FILE = "features/phase2_4/rna_enhanced_features.csv"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# === Utilities === #
def pseudo_torsion(p1, p2, p3, p4):
    try:
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        x = np.dot(n1, n2)
        y = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))
        angle = np.degrees(np.arctan2(y, x))
        return angle if np.isfinite(angle) else None
    except Exception:
        return None

def detect_motifs(torsions):
    motifs = {
        "a_minor": 0,
        "coaxial_helix": 0,
        "kissing_loop": 0
    }
    for t in torsions:
        if -30 <= t <= 30:
            motifs["a_minor"] += 1
        elif 60 <= t <= 120:
            motifs["coaxial_helix"] += 1
        elif -170 <= t <= -110:
            motifs["kissing_loop"] += 1
    return motifs

# === Load Metadata === #
print("\n🔍 Loading sequence + structure metadata...")
meta = pd.read_csv(META_FILE)
df_feat = pd.read_csv(RNA_FEAT_FILE)

# === Geometry & Motif Feature Extraction === #
print("\n🧬 Extracting torsion geometry, pucker state, and motif counts...")
residue_records = []
torsion_series = []
cluster_records = []
summary_records = []

for _, row in tqdm(meta.iterrows(), total=len(meta)):
    tid = row["target_id"]
    conf = row["conformation"]
    coord_file = os.path.join(COORD_PATH, f"{tid}_conf{conf}_clean.npy")

    if not os.path.exists(coord_file):
        continue

    coords = np.load(coord_file)
    n = coords.shape[0]

    torsions = [np.nan] * n
    puckers = ["unknown"] * n

    valid_torsions = []

    for i in range(n - 3):
        t = pseudo_torsion(coords[i], coords[i+1], coords[i+2], coords[i+3])
        if t is not None:
            torsions[i+1] = t
            puckers[i+1] = "C3'-endo" if t < 120 else "C2'-endo"
            valid_torsions.append(t)

    # Per-residue records
    for resid in range(n):
        residue_records.append({
            "target_id": tid,
            "conformation": conf,
            "resid": resid + 1,
            "pseudo_torsion_angle": torsions[resid],
            "pucker_state": puckers[resid]
        })

    # Per-conformation summary records
    if valid_torsions:
        motifs = detect_motifs(valid_torsions)
        summary_records.append({
            "target_id": tid,
            "conformation": conf,
            "pseudo_torsion_mean": np.mean(valid_torsions),
            "pseudo_torsion_std": np.std(valid_torsions),
            "a_minor_count": motifs["a_minor"],
            "coaxial_helix_count": motifs["coaxial_helix"],
            "kissing_loop_count": motifs["kissing_loop"]
        })

    # Clustering features
    if len(valid_torsions) >= 10:
        series = valid_torsions[:10]
        torsion_series.append((tid, conf, series))

# === Optional: Cluster Torsion Patterns === #
print("\n Performing torsion pattern clustering...")
if torsion_series:
    ids, confs, series = zip(*torsion_series)
    X = np.vstack(series)
    X = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
    labels = kmeans.labels_

    for tid, conf, label in zip(ids, confs, labels):
        cluster_records.append({
            "target_id": tid,
            "conformation": conf,
            "torsion_cluster": int(label)
        })

# === Final Merge === #
df_residue = pd.DataFrame(residue_records)
df_summary = pd.DataFrame(summary_records)
df_clusters = pd.DataFrame(cluster_records)

df_merged = df_residue.merge(df_summary, on=["target_id", "conformation"], how="left")
df_merged = df_merged.merge(df_clusters, on=["target_id", "conformation"], how="left")

# === Save === #
df_merged.to_csv(OUT_FILE, index=False)
print(f"\n Per-residue RNA geometry + clustering + motif counts saved to `{OUT_FILE}` — shape: {df_merged.shape}")
