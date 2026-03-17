import os
import pandas as pd
import numpy as np

# === CONFIG === #
FEATURE_PATHS = {
 "msa_graph_residue": "features/msa_graph_features_residue.csv",
 "rna_aware": "features/phase2_2/rna_aware_features.csv",
 "rna_enhanced": "features/phase2_4/rna_enhanced_features.csv",
 "structure_features": "features/phase2_3/structure_features.csv",
 "structure_summary": "features/phase2_3/structure_features_summary.csv",
 "torsion": "features/phase1_4/backbone_torsions.csv",
 "coord_meta": "coords/coord_metadata.csv",
 "surrogate": "features/phase2_2/rna_surrogate_preds.csv"
}
MERGED_SAVE_PATH = "features/final_merged_features.csv"
os.makedirs(os.path.dirname(MERGED_SAVE_PATH), exist_ok=True)

# === Load Feature Files === #
print("\n Loading feature files...")
df_msa = pd.read_csv(FEATURE_PATHS["msa_graph_residue"])
df_rna_aware = pd.read_csv(FEATURE_PATHS["rna_aware"])
df_rna_enh = pd.read_csv(FEATURE_PATHS["rna_enhanced"])
df_struct = pd.read_csv(FEATURE_PATHS["structure_features"])
df_summary = pd.read_csv(FEATURE_PATHS["structure_summary"])
df_torsion = pd.read_csv(FEATURE_PATHS["torsion"])
df_meta = pd.read_csv(FEATURE_PATHS["coord_meta"])
df_surrogate = pd.read_csv(FEATURE_PATHS["surrogate"]) if os.path.exists(FEATURE_PATHS["surrogate"]) else pd.DataFrame()

# === Normalize Keys === #
def normalize(df, keys):
 for k in keys:
 if k in df.columns:
 if k == "resid":
 df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0).astype(int)
 else:
 df[k] = df[k].astype(str).str.strip()

normalize(df_msa, ["target_id", "resid"])
normalize(df_rna_aware, ["target_id", "conformation", "resid"])
normalize(df_rna_enh, ["target_id", "conformation", "resid"])
normalize(df_struct, ["target_id", "conformation", "resid"])
normalize(df_summary, ["target_id", "conformation"])
normalize(df_torsion, ["target_id", "resid"])
df_torsion["conformation"] = "0"
normalize(df_torsion, ["target_id", "conformation", "resid"])

if not df_surrogate.empty:
 normalize(df_surrogate, ["target_id", "conformation", "resid"])

# === Deduplicate === #
def deduplicate(df, keys, name):
 dupes = df.duplicated(subset=keys, keep=False)
 if dupes.any():
 print(f" {name}: Found {dupes.sum()} duplicated rows. Resolving...")
 num_cols = df.select_dtypes(include=[np.number]).columns.difference(keys)
 cat_cols = df.select_dtypes(include="object").columns.difference(keys)
 agg_dict = {col: "mean" for col in num_cols}
 for col in cat_cols:
 agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
 df = df.groupby(keys, as_index=False).agg(agg_dict)
 return df

df_rna_enh = deduplicate(df_rna_enh, ["target_id", "conformation", "resid"], "rna_enhanced")
df_struct = deduplicate(df_struct, ["target_id", "conformation", "resid"], "structure_features")
df_rna_aware = deduplicate(df_rna_aware, ["target_id", "conformation", "resid"], "rna_aware")
df_msa = deduplicate(df_msa, ["target_id", "resid"], "msa_graph")
df_torsion = deduplicate(df_torsion, ["target_id", "conformation", "resid"], "torsions")
if not df_surrogate.empty:
 df_surrogate = deduplicate(df_surrogate, ["target_id", "conformation", "resid"], "surrogate")

# === Merge Features === #
print("\n Merging features...")
df = df_rna_enh.merge(df_struct, on=["target_id", "conformation", "resid"], how="left")
df = df.merge(df_rna_aware, on=["target_id", "conformation", "resid"], how="left")
df = df.merge(df_msa, on=["target_id", "resid"], how="left")
df = df.merge(df_summary, on=["target_id", "conformation"], how="left")
df = df.merge(df_torsion, on=["target_id", "conformation", "resid"], how="left")
if not df_surrogate.empty:
 df = df.merge(df_surrogate, on=["target_id", "conformation", "resid"], how="left")

print(f" After merging: {df.shape}")

# === Flags & Metadata === #
print("\n Adding flags...")
df["has_torsion"] = df["pseudo_torsion_angle"].notna().astype(int)
df["has_motif"] = df["motif_label"].notna().astype(int)
df["has_clash_info"] = df["clash_score"].notna().astype(int)
df["has_curvature"] = df["curvature"].notna().astype(int)

# === Feature Completeness Score === #
ignore_cols = ["target_id", "conformation", "resid", "resname", "motif_label", "prev_motif", "next_motif", "pucker_state"]
feature_cols = df.columns.difference(ignore_cols)
df["non_nan_fraction"] = df[feature_cols].notna().sum(axis=1) / len(feature_cols)

# === Merge Coord Metadata Flags (augmentation info) === #
print("\n Merging metadata flags from coord_metadata...")
normalize(df_meta, ["target_id", "conformation"])
df = df.merge(df_meta[["target_id", "conformation", "version", "is_valid"]], on=["target_id", "conformation"], how="left")


# === Fill NaNs === #
print("\n Filling NaNs...")
nan_cols = df.columns[df.isna().any()]
cat_cols = ["motif_label", "pucker_state", "prev_motif", "next_motif"]
num_cols = [col for col in nan_cols if col not in cat_cols]
df[num_cols] = df[num_cols].fillna(0.0)
df[cat_cols] = df[cat_cols].fillna("none")

# === Save === #
print("\n Saving final merged feature set...")
df.to_csv(MERGED_SAVE_PATH, index=False)
print(f" Done. Saved to `{MERGED_SAVE_PATH}` — shape: {df.shape}")
