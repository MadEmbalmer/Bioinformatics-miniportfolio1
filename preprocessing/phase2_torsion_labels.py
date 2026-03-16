import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# === CONFIG === #
MERGED_FILE = "features/final_merged_features.csv"
SAVE_FILE = "features/phase2_6/multitask_labels.csv"
CHUNK_SIZE = 10_000_000  # Lower this if you hit memory pressure
MAX_WORKERS = os.cpu_count() or 8

TORSION_COLS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)


def process_chunk(df):
    df["resid"] = pd.to_numeric(df["resid"], errors="coerce").fillna(0).astype(int)
    df["target_id"] = df["target_id"].astype(str).str.strip()
    df["conformation"] = df["conformation"].astype(str).str.strip()
    df["version"] = df.get("version", "clean").fillna("clean").astype(str)
    df["paired_with"] = pd.to_numeric(df.get("paired_with", np.nan), errors="coerce")
    df["basepair_distance"] = (df["resid"] - df["paired_with"]).abs().fillna(0).astype(np.float32)

    df_out = pd.DataFrame({
        "target_id": df["target_id"],
        "conformation": df["conformation"],
        "resid": df["resid"],
        "version": df["version"],
        "pseudo_alpha": pd.to_numeric(df.get("pseudo_torsion_angle", np.nan), errors="coerce").astype(np.float32),
        "torsion_angle": pd.to_numeric(df.get("torsion_angle", np.nan), errors="coerce").astype(np.float32),
        "clash_score": pd.to_numeric(df.get("clash_score", 0.0), errors="coerce").astype(np.float32),
        "curvature": pd.to_numeric(df.get("curvature", 0.0), errors="coerce").astype(np.float32),
        "entropy": pd.to_numeric(df.get("sequence_entropy", 0.0), errors="coerce").astype(np.float32),
        "basepair_distance": df["basepair_distance"]
    })

    for torsion in TORSION_COLS:
        if torsion in df.columns:
            df_out[torsion] = pd.to_numeric(df[torsion], errors="coerce").astype(np.float32)

    return df_out.dropna(subset=["alpha", "beta"], how="any")


def main():
    print("📥 Streaming and processing in parallel chunks...")
    first_chunk = True

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, chunk in enumerate(pd.read_csv(MERGED_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
            print(f"🔄 Scheduling chunk {i + 1} for processing...")
            futures.append(executor.submit(process_chunk, chunk))

        for i, future in enumerate(futures):
            df_result = future.result()
            df_result.to_csv(SAVE_FILE, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            print(f"✅ Wrote chunk {i + 1} — rows: {df_result.shape[0]}")
            first_chunk = False

    print(f"\n✅ Done. Saved multi-task labels to `{SAVE_FILE}`")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Required for Windows multiprocessing safety
    main()
