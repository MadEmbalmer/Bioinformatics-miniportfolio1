import pandas as pd
import os

# Input file
final_merged_file = "features/final_merged_features.csv"
output_file = "features/torsion_labels.csv"

# Load merged features
print(" Loading final merged features...")
df = pd.read_csv(final_merged_file)

# Ensure correct columns exist
required_cols = ["target_id", "resid", "pseudo_torsion_angle", "torsion_angle"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Construct ID and extract
df["ID"] = df["target_id"] + "_" + df["resid"].astype(str)
final = df[["ID", "pseudo_torsion_angle", "torsion_angle"]].rename(
    columns={
        "pseudo_torsion_angle": "alpha",
        "torsion_angle": "beta"
    }
)

# Save
os.makedirs("features", exist_ok=True)
final.to_csv(output_file, index=False)
print(f" Torsion labels saved to: {output_file}")