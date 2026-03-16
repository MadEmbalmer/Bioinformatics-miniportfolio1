import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import joblib

# === CONFIG === #
FEATURE_FILE = "features/final_merged_features.csv"
SAVE_DIR = "features/phase3"
ENCODER_WEIGHTS = os.path.join(SAVE_DIR, "encoder.pt")
CHUNK_SIZE = 15_000_000
N_COMPONENTS = 50
LATENT_DIM = 8
EPOCHS = 50
BATCH_SIZE = 512

os.makedirs(SAVE_DIR, exist_ok=True)

# === MODEL DEFINITIONS === #
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

# === Setup === #
print("📥 Streaming data for scaling and PCA...")
id_cols = ["target_id", "conformation", "resid", "pucker_state"]
scaler = StandardScaler()
pca = IncrementalPCA(n_components=N_COMPONENTS)

# === First pass: Fit scaler and PCA === #
for chunk in pd.read_csv(FEATURE_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    numeric = chunk.drop(columns=[col for col in id_cols if col in chunk.columns], errors="ignore")
    numeric = numeric.select_dtypes(include=[np.number]).fillna(0.0)
    scaler.partial_fit(numeric)
    scaled = scaler.transform(numeric)
    pca.partial_fit(scaled)

joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
joblib.dump(pca, os.path.join(SAVE_DIR, "pca.pkl"))
print("✅ Scaler and PCA saved.")

# === Collect data for encoder training === #
X_train = []
for chunk in pd.read_csv(FEATURE_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    numeric = chunk.drop(columns=[col for col in id_cols if col in chunk.columns], errors="ignore")
    numeric = numeric.select_dtypes(include=[np.number]).fillna(0.0)
    scaled = scaler.transform(numeric)
    reduced = pca.transform(scaled)
    X_train.append(torch.tensor(reduced, dtype=torch.float32))

X_train = torch.cat(X_train, dim=0)
train_loader = torch.utils.data.DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)

# === Train Encoder === #
print("🧠 Training encoder...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_dim=N_COMPONENTS, latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}: Loss = {total_loss / len(train_loader):.6f}")

torch.save(model.encoder.state_dict(), ENCODER_WEIGHTS)
print(f"✅ Encoder weights saved to {ENCODER_WEIGHTS}")

# === Inference: Extract latent vectors === #
print("🔁 Encoding all features...")
model.eval()
z_all, id_all = [], []
encoder = model.encoder
for chunk in pd.read_csv(FEATURE_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    ids = chunk[id_cols]
    numeric = chunk.drop(columns=[col for col in id_cols if col in chunk.columns], errors="ignore")
    numeric = numeric.select_dtypes(include=[np.number]).fillna(0.0)
    scaled = scaler.transform(numeric)
    reduced = pca.transform(scaled)
    X_tensor = torch.tensor(reduced, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_chunk = encoder(X_tensor).cpu().numpy()
    z_all.append(z_chunk)
    id_all.append(ids)

z_np = np.vstack(z_all)
df_ids = pd.concat(id_all, ignore_index=True)
np.save(os.path.join(SAVE_DIR, "latent_vectors.npy"), z_np)

# === Clustering (Residue-Level) === #
print("🔍 Clustering residues (sampling for silhouette)...")
sample_idx = np.random.choice(len(z_np), size=min(100_000, len(z_np)), replace=False)
z_sample = z_np[sample_idx]

sil_scores = []
for k in range(2, 10):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096).fit(z_sample)
    sil_scores.append(silhouette_score(z_sample, km.labels_))

best_k = np.argmax(sil_scores) + 2
print(f"   ➔ Best residue-level k (sampled): {best_k}")
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=4096).fit(z_np)
clusters = kmeans.labels_
centroids = kmeans.cluster_centers_

dists = cdist(z_np, centroids)
min_dists = dists[np.arange(len(dists)), clusters]

df_ids["residue_cluster"] = clusters
df_ids["diversity_score"] = min_dists
df_ids.to_csv(os.path.join(SAVE_DIR, "clustered_features.csv"), index=False)

pd.DataFrame({"cluster": range(best_k), "count": np.bincount(clusters)}).to_csv(
    os.path.join(SAVE_DIR, "cluster_summary.csv"), index=False)

# === RNA-Level Clustering === #
print("🏷️ Clustering RNAs...")
df_rna = df_ids.groupby("target_id").mean(numeric_only=True)
rna_scaled = StandardScaler().fit_transform(df_rna)

sil_rna = []
for k in range(2, 10):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096).fit(rna_scaled)
    sil_rna.append(silhouette_score(rna_scaled, km.labels_))

best_k_rna = np.argmax(sil_rna) + 2
print(f"   ➔ Best RNA-level k: {best_k_rna}")
rna_kmeans = MiniBatchKMeans(n_clusters=best_k_rna, random_state=42, batch_size=4096).fit(rna_scaled)
df_rna["rna_cluster"] = rna_kmeans.labels_

# Save RNA cluster IDs
rna_cluster_df = df_rna.reset_index()[["target_id", "rna_cluster"]]
rna_cluster_df.to_csv(os.path.join(SAVE_DIR, "rna_clusters.csv"), index=False)

# === Prototypes === #
print("🐜 Finding prototypes...")
rna_centroids = rna_kmeans.cluster_centers_
rna_dists = cdist(rna_scaled, rna_centroids)
prototypes = df_rna.reset_index().iloc[np.argmin(rna_dists, axis=0)][["target_id", "rna_cluster"]]
prototypes.to_csv(os.path.join(SAVE_DIR, "prototypes.csv"), index=False)

print("\n🚀 DONE: All outputs saved to", SAVE_DIR)