import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === CONFIG === #
META_FILE = "coords/coord_metadata.csv"
COORD_PATH = "coords"
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
    motifs = []
    for t in torsions:
        if t is None or np.isnan(t):
            motifs.append("none")
        elif -30 <= t <= 30:
            motifs.append("a_minor")
        elif 60 <= t <= 120:
            motifs.append("coaxial_helix")
        elif -170 <= t <= -110:
            motifs.append("kissing_loop")
        else:
            motifs.append("other")
    return motifs

def encode_motif(motif):
    return {
        "a_minor": 0,
        "coaxial_helix": 1,
        "kissing_loop": 2,
        "other": 3,
        "none": 4
    }.get(motif, 4)

def get_loop_lengths(motifs):
    loop_lengths = [1] * len(motifs)
    current = 0
    for i in range(1, len(motifs)):
        if motifs[i] == motifs[i - 1]:
            current += 1
        else:
            current = 1
        loop_lengths[i] = current
    return loop_lengths

# === Load Metadata === #
print("\n🔍 Loading coordinate metadata...")
meta = pd.read_csv(META_FILE)

# === Feature Extraction === #
print("\n🧬 Extracting enhanced structural features...")
residue_records = []
summary_records = []
torsion_series = []
cluster_records = []

for _, row in tqdm(meta.iterrows(), total=len(meta)):
    tid = str(row["target_id"]).strip()
    conf = str(row["conformation"]).strip()
    coord_file = os.path.join(COORD_PATH, f"{tid}_conf{conf}_clean.npy")

    if not os.path.exists(coord_file):
        continue

    try:
        coords = np.load(coord_file)
        n = len(coords)
        torsions = [np.nan] * n
        puckers = ["unknown"] * n
        valid_torsions = []

        for i in range(n - 3):
            t = pseudo_torsion(coords[i], coords[i + 1], coords[i + 2], coords[i + 3])
            if t is not None:
                torsions[i + 1] = t
                puckers[i + 1] = "C3'-endo" if t < 120 else "C2'-endo"
                valid_torsions.append(t)

        motif_labels = detect_motifs(torsions)
        motif_encoded = [encode_motif(m) for m in motif_labels]
        motif_prev = ["none"] + motif_labels[:-1]
        motif_next = motif_labels[1:] + ["none"]
        loop_lengths = get_loop_lengths(motif_labels)

        for resid in range(n):
            residue_records.append({
                "target_id": tid,
                "conformation": conf,
                "resid": resid + 1,
                "pseudo_torsion_angle": torsions[resid],
                "pucker_state": puckers[resid],
                "motif_label": motif_labels[resid],
                "motif_index": motif_encoded[resid],
                "prev_motif": motif_prev[resid],
                "next_motif": motif_next[resid],
                "loop_length": loop_lengths[resid]
            })

        if valid_torsions:
            motifs = detect_motifs(valid_torsions)
            summary_records.append({
                "target_id": tid,
                "conformation": conf,
                "pseudo_torsion_mean": np.mean(valid_torsions),
                "pseudo_torsion_std": np.std(valid_torsions),
                "a_minor_count": motifs.count("a_minor"),
                "coaxial_helix_count": motifs.count("coaxial_helix"),
                "kissing_loop_count": motifs.count("kissing_loop")
            })

        if len(valid_torsions) >= 10:
            series = valid_torsions[:10]
            if len(series) == 10 and all(np.isfinite(series)):
                torsion_series.append((tid, conf, series))

    except Exception as e:
        print(f"⚠️ Error with {tid} conf {conf}: {e}")
        continue

# === Clustering === #
print("\n🧪 Clustering torsion series...")
if torsion_series:
    ids, confs, series = zip(*torsion_series)
    X = np.array(series)
    X = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
    labels = kmeans.labels_

    for tid, conf, label in zip(ids, confs, labels):
        cluster_records.append({
            "target_id": tid,
            "conformation": conf,
            "torsion_cluster": int(label)
        })

# === Merge and Save === #
df_residue = pd.DataFrame(residue_records)
df_summary = pd.DataFrame(summary_records)
df_clusters = pd.DataFrame(cluster_records)

print("\n📎 Normalizing columns before merging...")
for df in [df_residue, df_summary, df_clusters]:
    df["target_id"] = df["target_id"].astype(str).str.strip()
    df["conformation"] = df["conformation"].astype(str).str.strip()
    if "resid" in df.columns:
        df["resid"] = df["resid"].astype(int)

print("\n📆 Merging features and saving...")
df_merged = df_residue.merge(df_summary, on=["target_id", "conformation"], how="left")
df_merged = df_merged.merge(df_clusters, on=["target_id", "conformation"], how="left")

df_merged.to_csv(OUT_FILE, index=False)
print(f"\n✅ Saved to `{OUT_FILE}` — shape: {df_merged.shape}")
