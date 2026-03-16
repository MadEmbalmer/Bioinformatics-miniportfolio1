import os
import pandas as pd
import numpy as np
from collections import Counter
from Bio import SeqIO
from scipy.stats import entropy

# === CONFIG === #
SEQ_FILE = "phase1_cleaned_sequences.csv"
MSA_DIR = "data/MSA"  # directory containing per-sequence MSA fasta files
SAVE_DIR = "features"
MSA_FEATURE_FILE = os.path.join(SAVE_DIR, "msa_pssm_conservation.npz")
SKIPPED_LOG = os.path.join(SAVE_DIR, "skipped_msa_targets.csv")

ALLOWED_BASES = ['A', 'C', 'G', 'U']
MIN_MSA_DEPTH = 3

# === Ensure directories exist === #
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Sequences === #
print("📄 Loading cleaned sequences from Phase 1.1...")
df = pd.read_csv(SEQ_FILE)

if "target_id" not in df.columns or "clean_sequence" not in df.columns:
    raise ValueError("❌ 'target_id' and 'clean_sequence' columns are required in the CSV.")

target_ids = df["target_id"].tolist()
clean_seqs = dict(zip(df["target_id"], df["clean_sequence"]))

# === Helper: Frequency matrix + entropy + conservation === #
def compute_pssm_entropy_conservation(msa_seqs, length):
    base_idx = {base: i for i, base in enumerate(ALLOWED_BASES)}
    freq_matrix = np.zeros((length, len(ALLOWED_BASES)), dtype=np.float32)
    entropy_vector = np.zeros(length, dtype=np.float32)
    conservation_vector = np.zeros(length, dtype=np.float32)

    for i in range(length):
        col = [seq[i] for seq in msa_seqs if i < len(seq)]
        counts = Counter(col)
        total = sum(counts[b] for b in ALLOWED_BASES)

        probs = []
        for base in ALLOWED_BASES:
            p = counts[base] / total if total > 0 else 0.0
            freq_matrix[i][base_idx[base]] = p
            probs.append(p)

        if total > 0:
            ent = entropy(probs, base=2)
            entropy_vector[i] = ent
            conservation_vector[i] = 1.0 - (ent / np.log2(len(ALLOWED_BASES)))
        else:
            entropy_vector[i] = 0.0
            conservation_vector[i] = 0.0

    return freq_matrix, entropy_vector, conservation_vector

# === Helper: Fallback PSSM === #
def generate_fallback_pssm(ref_seq):
    base_idx = {base: i for i, base in enumerate(ALLOWED_BASES)}
    fallback = np.zeros((len(ref_seq), len(ALLOWED_BASES)), dtype=np.float32)
    for i, base in enumerate(ref_seq):
        if base in base_idx:
            fallback[i][base_idx[base]] = 1.0
    entropy_vec = np.zeros(len(ref_seq), dtype=np.float32)
    conservation_vec = np.ones(len(ref_seq), dtype=np.float32)
    return fallback, entropy_vec, conservation_vec

# === Main Extraction Loop with Hybrid Strategy === #
msa_feature_dict = {}
skipped_targets = []

print(f"🔬 Extracting PSSM + entropy + conservation from {len(target_ids)} MSA files...")

for tid in target_ids:
    ref_seq = clean_seqs[tid]
    seq_len = len(ref_seq)

    # Option 1: Try strict match (chain-specific)
    strict_msa_path = os.path.join(MSA_DIR, f"{tid}.MSA.fasta")
    msa_path = strict_msa_path

    if not os.path.exists(msa_path):
        # Option 2: Try fallback to PDB ID only (relaxed match)
        pdb_id = tid.split("_")[0]
        relaxed_msa_path = os.path.join(MSA_DIR, f"{pdb_id}.MSA.fasta")
        if os.path.exists(relaxed_msa_path):
            msa_path = relaxed_msa_path
        else:
            print(f"⚠️ No MSA file for {tid} or {pdb_id}, using fallback PSSM.")
            fallback_pssm, ent, cons = generate_fallback_pssm(ref_seq)
            msa_feature_dict[tid] = np.concatenate([fallback_pssm, ent[:, None], cons[:, None]], axis=1)
            skipped_targets.append({"target_id": tid, "reason": "no_msa_fallback"})
            continue

    try:
        msa_seqs = [
            str(record.seq).upper().replace('T', 'U')
            for record in SeqIO.parse(msa_path, "fasta")
        ]

        msa_seqs = [seq for seq in msa_seqs if set(seq).issubset(set(ALLOWED_BASES + ['-']))]
        msa_seqs = [seq[:seq_len].ljust(seq_len, '-') for seq in msa_seqs if len(seq) >= seq_len // 2]
        msa_seqs = [seq for seq in msa_seqs if len(seq) == seq_len]

        if len(msa_seqs) < MIN_MSA_DEPTH:
            print(f"⚠️ Shallow MSA ({len(msa_seqs)}) for {tid}, using fallback PSSM.")
            fallback_pssm, ent, cons = generate_fallback_pssm(ref_seq)
            msa_feature_dict[tid] = np.concatenate([fallback_pssm, ent[:, None], cons[:, None]], axis=1)
            skipped_targets.append({"target_id": tid, "reason": f"shallow_msa_fallback_{len(msa_seqs)}"})
            continue

        pssm, entropy_vec, conservation_vec = compute_pssm_entropy_conservation(msa_seqs, length=seq_len)
        msa_feature_dict[tid] = np.concatenate([pssm, entropy_vec[:, None], conservation_vec[:, None]], axis=1)

    except Exception as e:
        print(f"❌ Error processing {tid}: {e}")
        skipped_targets.append({"target_id": tid, "reason": str(e)})
        fallback_pssm, ent, cons = generate_fallback_pssm(ref_seq)
        msa_feature_dict[tid] = np.concatenate([fallback_pssm, ent[:, None], cons[:, None]], axis=1)

# === Save Outputs === #
if msa_feature_dict:
    np.savez_compressed(MSA_FEATURE_FILE, **msa_feature_dict)
    print(f"\n✅ MSA PSSM + entropy + conservation features saved to {MSA_FEATURE_FILE} for {len(msa_feature_dict)} sequences.")
else:
    print("❌ No valid MSA features were extracted. Please check input files.")

if skipped_targets:
    pd.DataFrame(skipped_targets).to_csv(SKIPPED_LOG, index=False)
    print(f"📄 Logged {len(skipped_targets)} fallback/skipped targets to {SKIPPED_LOG}")
