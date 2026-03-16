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
ALLOWED_BASES = ['A', 'C', 'G', 'U']

# === Ensure directories exist === #
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Sequences === #
print("\U0001F4C4 Loading cleaned sequences from Phase 1.1...")
df = pd.read_csv(SEQ_FILE)

if "target_id" not in df.columns or "clean_sequence" not in df.columns:
    raise ValueError(" 'target_id' and 'clean_sequence' columns are required in the CSV.")

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

# === Main Extraction Loop === #
msa_feature_dict = {}

print(f"\U0001F9EC Extracting PSSM + entropy + conservation from {len(target_ids)} MSA files...")

for tid in target_ids:
    msa_path = os.path.join(MSA_DIR, f"{tid}.MSA.fasta")

    if not os.path.exists(msa_path):
        print(f" Missing MSA file for target_id: {tid}, skipping.")
        continue

    try:
        msa_seqs = [str(record.seq).upper().replace('T', 'U') for record in SeqIO.parse(msa_path, "fasta")]
        msa_seqs = [seq for seq in msa_seqs if set(seq).issubset(set(ALLOWED_BASES + ['-']))]

        ref_seq = clean_seqs[tid]
        seq_len = len(ref_seq)

        msa_seqs = [seq.ljust(seq_len, '-')[:seq_len] for seq in msa_seqs]
        msa_seqs = [seq for seq in msa_seqs if len(seq) == seq_len]

        if len(msa_seqs) < 1:
            print(f" No aligned sequences for {tid}, skipping.")
            continue

        pssm, entropy_vec, conservation_vec = compute_pssm_entropy_conservation(msa_seqs, length=seq_len)
        msa_feature_dict[tid] = np.concatenate([pssm, entropy_vec[:, None], conservation_vec[:, None]], axis=1)

    except Exception as e:
        print(f" Error processing {tid}: {e}")
        continue

# === Save all to compressed .npz file === #
if msa_feature_dict:
    np.savez_compressed(MSA_FEATURE_FILE, **msa_feature_dict)
    print(f"\n MSA PSSM + entropy + conservation features saved to {MSA_FEATURE_FILE} for {len(msa_feature_dict)} sequences.")
else:
    print(" No valid MSA features were extracted. Please check input files.")
