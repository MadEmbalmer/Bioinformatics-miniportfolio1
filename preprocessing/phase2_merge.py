import os
import pandas as pd

# === CONFIG === #
FEATURE_PATHS = {
    "msa_graph_residue": "features/msa_graph_features_residue.csv",   # Per-residue MSA + graph features
    "rna_aware": "features/phase2_2/rna_aware_features.csv",          # Per-residue RNAfold & secondary structure
    "rna_enhanced": "features/phase2_4/rna_enhanced_features.csv",    # Pseudo-torsions, motifs, clustering
    "structure_features": "features/phase2_3/structure_features.csv" # Bond angles, torsions, steric clashes
}
MERGED_SAVE_PATH = "features/final_merged_features.csv"
os.makedirs(os.path.dirname(MERGED_SAVE_PATH), exist_ok=True)

# === Load Data === #
print(" Loading feature files...")
df_msa_residue = pd.read_csv(FEATURE_PATHS["msa_graph_residue"])
df_rna_aware = pd.read_csv(FEATURE_PATHS["rna_aware"])
df_rna_enhanced = pd.read_csv(FEATURE_PATHS["rna_enhanced"])
df_structure = pd.read_csv(FEATURE_PATHS["structure_features"])

print(f" - Loaded MSA Graph Residue Features: {df_msa_residue.shape}")
print(f" - Loaded RNA-aware Features: {df_rna_aware.shape}")
print(f" - Loaded RNA-enhanced Features (motifs): {df_rna_enhanced.shape}")
print(f" - Loaded Structure Features (geometry): {df_structure.shape}")

# === Merge per-residue features === #
print(" Merging per-residue features...")

# Merge in safe order: Enhanced → Structure → RNA-aware → MSA
df = df_rna_enhanced.merge(df_structure, on=["target_id", "conformation", "resid"], how="inner")
print(f"   After merging structure features: {df.shape}")

df = df.merge(df_rna_aware, on=["target_id", "conformation", "resid"], how="inner")
print(f"   After merging RNA-aware features: {df.shape}")

df = df.merge(df_msa_residue, on=["target_id", "resid"], how="inner")
print(f"   After merging MSA-graph features: {df.shape}")

# === Save === #
print("\n Saving final merged features...")
df.to_csv(MERGED_SAVE_PATH, index=False)
print(f" Final merged features saved to `{MERGED_SAVE_PATH}` — final shape: {df.shape}")
