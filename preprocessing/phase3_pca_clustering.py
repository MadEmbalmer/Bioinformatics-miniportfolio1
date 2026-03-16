import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# === CONFIG === #
FEATURE_FILE = "features/final_merged_features.csv"
SAVE_DIR = "features/phase3"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Features === #
print(" Loading merged features...")
df = pd.read_csv(FEATURE_FILE)

# === Drop Identifiers === #
print(" Cleaning data (dropping non-numeric columns)...")
drop_cols = ["target_id", "conformation", "resid", "pucker_state"]  # identifiers
X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Only keep numeric features
X = X.select_dtypes(include=[np.number])

print(f" - Retained {X.shape[1]} numeric features.")

# === Handle Missing Values === #
print(" Filling missing values with feature-wise mean...")
imputer = SimpleImputer(strategy="mean")
X_filled = imputer.fit_transform(X)

# === Standardize === #
print(" Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# === PCA === #
print(" Running PCA (2 components)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Save PCA coordinates
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["target_id"] = df["target_id"]
pca_df["conformation"] = df["conformation"]
pca_df["resid"] = df["resid"]
pca_df.to_csv(os.path.join(SAVE_DIR, "pca_coordinates.csv"), index=False)

print(f" Saved PCA coordinates for {len(pca_df)} residues.")

# === KMeans Clustering === #
print(" Clustering in full feature space with KMeans (k=5)...")
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca_df["cluster"] = clusters
pca_df.to_csv(os.path.join(SAVE_DIR, "pca_with_clusters.csv"), index=False)

print(f" Saved cluster assignments.")

# === Visualization === #
print(" Plotting PCA clusters...")
plt.figure(figsize=(10,8))
scatter = plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["cluster"], cmap="viridis", s=10)
plt.title("Residue PCA Projection with Clusters", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pca_clusters_plot.png"))
plt.show()

print("\n Phase 3 Starter Completed!")
