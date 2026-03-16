import os
import pandas as pd
import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
import numpy as np
import traceback

# === CONFIG === #
SEQ_FILE = "phase1_cleaned_sequences.csv"
PDB_DIR = "pdb_files"
FAILED_LOG = os.path.join(PDB_DIR, "failed_downloads.txt")
OUTPUT_FILE = "features/phase1_4/backbone_torsions.csv"
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# === Globals === #
parser = PDBParser(QUIET=True)
failed_cache = set()

# === Download Functions === #
def log_failure(key: str):
    if key not in failed_cache:
        with open(FAILED_LOG, "a") as f:
            f.write(f"{key}\n")
            f.flush()
        failed_cache.add(key)

def download_pdb(pdb_id: str, chain_id: str, save_path: str, key: str) -> bool:
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"❌ {pdb_id}: HTTP {r.status_code}")
            log_failure(key)
            return False

        lines = r.text.splitlines()
        chain_lines = [line for line in lines if line.startswith("ATOM") and line[21:22].strip() == chain_id]
        if not chain_lines:
            print(f"⚠️ {pdb_id} chain {chain_id} not found.")
            log_failure(key)
            return False

        with open(save_path, "w") as f:
            f.write("\n".join(chain_lines) + "\n")
        print(f"✅ Saved {save_path}")
        return True

    except Exception as e:
        print(f"❌ Failed {pdb_id} chain {chain_id}: {e}")
        traceback.print_exc()
        log_failure(key)
        return False

def handle_download(pdb_id: str, chain_id: str, key: str) -> bool:
    filename = f"{pdb_id}_{chain_id}.pdb"
    filepath = os.path.join(PDB_DIR, filename)
    if not os.path.exists(filepath):
        return download_pdb(pdb_id, chain_id, filepath, key)
    else:
        print(f"✅ Already exists: {filename}")
        return True

# === Torsion Helper Functions === #
def get_atom_vector(residue, atom_name: str):
    return residue[atom_name].get_vector() if atom_name in residue else None

def compute_dihedral_safe(p1, p2, p3, p4):
    if any(x is None for x in [p1, p2, p3, p4]):
        return np.nan
    return np.degrees(calc_dihedral(p1, p2, p3, p4))

def process_structure(row: dict):
    tid = row["target_id"]
    pdb_id = row["pdb_id"]
    chain_id = row["chain_id"]
    pdb_file = os.path.join(PDB_DIR, f"{pdb_id}_{chain_id}.pdb")
    result_list = []

    if not os.path.isfile(pdb_file):
        print(f"⚠️ Missing PDB file for {tid}: {pdb_file}")
        return []

    try:
        structure = parser.get_structure(tid, pdb_file)
        chain = structure[0][chain_id]
        residues = [res for res in chain if res.id[0] == " "]

        for i in range(3, len(residues) - 3):
            r_prev1 = residues[i - 1]
            r0 = residues[i]
            r1 = residues[i + 1]

            atom_sets = {
                "alpha": [get_atom_vector(r_prev1, "P"), get_atom_vector(r0, "O5'"), get_atom_vector(r0, "C5'"), get_atom_vector(r0, "C4'")],
                "beta": [get_atom_vector(r0, "O5'"), get_atom_vector(r0, "C5'"), get_atom_vector(r0, "C4'"), get_atom_vector(r0, "C3'")],
                "gamma": [get_atom_vector(r0, "C5'"), get_atom_vector(r0, "C4'"), get_atom_vector(r0, "C3'"), get_atom_vector(r0, "O3'")],
                "delta": [get_atom_vector(r0, "C4'"), get_atom_vector(r0, "C3'"), get_atom_vector(r0, "O3'"), get_atom_vector(r1, "P")],
                "epsilon": [get_atom_vector(r0, "C3'"), get_atom_vector(r0, "O3'"), get_atom_vector(r1, "P"), get_atom_vector(r1, "O5'")],
                "zeta": [get_atom_vector(r0, "O3'"), get_atom_vector(r1, "P"), get_atom_vector(r1, "O5'"), get_atom_vector(r1, "C5'")],
                "chi": [
                    get_atom_vector(r0, "O4'"),
                    get_atom_vector(r0, "C1'"),
                    get_atom_vector(r0, "N9" if r0.resname in ["A", "G"] else "N1"),
                    get_atom_vector(r0, "C4" if r0.resname in ["A", "G"] else "C2")
                ]
            }

            torsions = {name: compute_dihedral_safe(*atoms) for name, atoms in atom_sets.items()}
            result_list.append({"target_id": tid, "resid": r0.id[1], **torsions})

        return result_list

    except Exception as e:
        print(f"❌ Failed processing {tid}: {e}")
        traceback.print_exc()
        return []

# === Main === #
if __name__ == "__main__":
    df = pd.read_csv(SEQ_FILE)

    # Extract PDB ID and chain_id more robustly
    df["pdb_id"] = df["target_id"].str.extract(r"([0-9A-Za-z]{4})", expand=False).str.upper()
    df["chain_id"] = df["target_id"].str.extract(r"_(.+)$", expand=False)
    df["key"] = df["pdb_id"] + "_" + df["chain_id"]

    targets = df[["pdb_id", "chain_id", "key"]].drop_duplicates()

    if os.path.exists(FAILED_LOG):
        with open(FAILED_LOG) as f:
            failed_cache = set(line.strip() for line in f)

    targets = targets[~targets["key"].isin(failed_cache)]

    print(f"\n📥 Downloading PDB chains for {len(targets)} unique targets...\n")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(handle_download, row.pdb_id, row.chain_id, row.key) for _, row in targets.iterrows()]
        for _ in as_completed(futures):
            pass
    print("\n✅ All downloads completed.")

    # Prepare for processing
    rows = df.to_dict(orient="records")

    print("\n📐 Extracting torsions...")
    with Pool(processes=cpu_count()) as pool:
        all_results = pool.map(process_structure, rows)

    results = [item for sublist in all_results for item in sublist if item]
    final_df = pd.DataFrame(results)

    if not final_df.empty:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Saved backbone torsions to {OUTPUT_FILE} — shape: {final_df.shape}")
    else:
        print("\n⚠️ No torsion data extracted.")
