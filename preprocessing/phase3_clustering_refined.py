import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# === CONFIGURATION === #
SAVE_DIR = "features/phase3"
LATENT_VECTORS_PATH = os.path.join(SAVE_DIR, "latent_vectors.npy")
ID_DATA_PATH = os.path.join(SAVE_DIR, "clustered_features.csv")

# Configure threading to prevent MKL memory leak warnings
NUM_THREADS = 12 # Adjust to match your CPU
BATCH_SIZE = max(6000, NUM_THREADS * 512)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

# === LOAD DATA === #
print(" Loading precomputed latent vectors and IDs...")
z_np = np.load(LATENT_VECTORS_PATH).astype(np.float64)
df_ids = pd.read_csv(ID_DATA_PATH).drop(columns=["residue_cluster", "diversity_score"], errors="ignore")

# === RESIDUE-LEVEL CLUSTERING === #
print(" Clustering residues (sampling for silhouette)...")
sample_idx = np.random.choice(len(z_np), size=min(100_000, len(z_np)), replace=False)
z_sample = z_np[sample_idx]

sil_scores = []
for k in range(2, 10):
 kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=BATCH_SIZE, n_init=5)
 labels = kmeans.fit_predict(z_sample)
 score = silhouette_score(z_sample, labels)
 sil_scores.append(score)

best_k = np.argmax(sil_scores) + 2
print(f" Best residue-level k (sampled): {best_k}")

kmeans_full = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=BATCH_SIZE, n_init=5)
clusters = kmeans_full.fit_predict(z_np)
centroids = kmeans_full.cluster_centers_

dists = cdist(z_np, centroids)
min_dists = dists[np.arange(len(dists)), clusters]

df_ids["residue_cluster"] = clusters
df_ids["diversity_score"] = min_dists
df_ids.to_csv(os.path.join(SAVE_DIR, "clustered_features.csv"), index=False)

pd.DataFrame({
 "cluster": np.arange(best_k),
 "count": np.bincount(clusters)
}).to_csv(os.path.join(SAVE_DIR, "cluster_summary.csv"), index=False)

# === RNA-LEVEL CLUSTERING === #
print(" Clustering RNAs...")
df_rna = df_ids.groupby("target_id").mean(numeric_only=True)
rna_scaled = StandardScaler().fit_transform(df_rna).astype(np.float64)

sil_rna = []
for k in range(2, 10):
 kmeans_rna = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=BATCH_SIZE, n_init=5)
 labels = kmeans_rna.fit_predict(rna_scaled)
 sil_rna.append(silhouette_score(rna_scaled, labels))

best_k_rna = np.argmax(sil_rna) + 2
print(f" Best RNA-level k: {best_k_rna}")

rna_kmeans = MiniBatchKMeans(n_clusters=best_k_rna, random_state=42, batch_size=BATCH_SIZE, n_init=5)
rna_labels = rna_kmeans.fit_predict(rna_scaled)
df_rna["rna_cluster"] = rna_labels

rna_cluster_df = df_rna.reset_index()[["target_id", "rna_cluster"]]
rna_cluster_df.to_csv(os.path.join(SAVE_DIR, "rna_clusters.csv"), index=False)

# === FIND PROTOTYPES === #
print(" Finding prototypes...")
rna_centroids = rna_kmeans.cluster_centers_
rna_dists = cdist(rna_scaled, rna_centroids)
prototype_idxs = np.argmin(rna_dists, axis=0)

prototypes = df_rna.reset_index().iloc[prototype_idxs][["target_id", "rna_cluster"]]
prototypes.to_csv(os.path.join(SAVE_DIR, "prototypes.csv"), index=False)

print("\n DONE: Clustering outputs saved to", SAVE_DIR)
