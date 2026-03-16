import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from tqdm import tqdm
from Bio.SeqUtils import gc_fraction
from scipy.stats import entropy as scipy_entropy
import subprocess
import tempfile

# === CONFIG === #
SEQ_FILE = "phase1_cleaned_sequences.csv"
SAVE_DIR = "features/phase2_2"
os.makedirs(SAVE_DIR, exist_ok=True)
DOTBRACKET_OUTPUT = os.path.join(SAVE_DIR, "dotbracket_strings.csv")
OUTPUT_FEATURES = os.path.join(SAVE_DIR, "rna_aware_features.csv")
VIENNARNA_EXECUTABLE = "RNAfold"  # assumes ViennaRNA installed (pip install viennarna)

# === Load sequences === #
print(" Loading cleaned sequences...")
df = pd.read_csv(SEQ_FILE)

if "target_id" not in df.columns or "clean_sequence" not in df.columns:
    raise ValueError("Missing required columns 'target_id' or 'clean_sequence' in input CSV.")

# === RNAfold Wrapper === #
def run_rnafold(sequence):
    try:
        with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as f:
            f.write(sequence + "\n")
            fname = f.name

        result = subprocess.run(
            [VIENNARNA_EXECUTABLE, "--noPS", fname],
            capture_output=True,
            text=True
        )
        os.unlink(fname)

        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            dot_bracket = lines[1].split()[0]
            mfe = float(lines[1].split("(")[-1].strip(")"))
            return dot_bracket, mfe
        else:
            return "." * len(sequence), 0.0
    except Exception as e:
        print(f" RNAfold failed for sequence: {e}")
        return "." * len(sequence), 0.0

# === Dot-Bracket Analysis === #
def parse_dot_bracket(dot, seq):
    stack = []
    pair_map = {}
    pair_types = []
    for i, char in enumerate(dot):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pair_map[i] = j
                pair_map[j] = i
                pair = ''.join(sorted([seq[i], seq[j]]))
                pair_types.append(pair)
    num_stems = len(pair_map) // 2
    num_loops = dot.count('.')
    pair_type_counts = Counter(pair_types)
    return pair_map, num_stems, num_loops, pair_type_counts

# === Feature functions === #
def base_pair_entropy(seq):
    counts = Counter(seq)
    probs = [v / len(seq) for v in counts.values()]
    return scipy_entropy(probs)

def sequence_entropy(seq):
    probs = [seq.count(b)/len(seq) for b in "ACGU" if b in seq]
    return scipy_entropy(probs)

def graph_topology_stats(seq):
    G = nx.Graph()
    for i, base in enumerate(seq):
        G.add_node(i, base=base)
        if i > 0:
            G.add_edge(i - 1, i)
    avg_deg = np.mean([deg for _, deg in G.degree()]) if len(G) > 0 else 0.0
    avg_clust = nx.average_clustering(G) if len(G) > 1 else 0.0
    return {
        "avg_degree": avg_deg,
        "avg_clustering": avg_clust
    }

# === Build Per-Residue Features === #
records = []
dotbracket_strings = []

print(" Extracting per-residue RNA-aware features...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    tid = row["target_id"]
    seq = row["clean_sequence"]

    if not isinstance(seq, str) or len(seq) < 5:
        print(f" Skipping very short or invalid sequence for {tid}")
        continue

    dot_bracket, mfe = run_rnafold(seq)
    pair_map, stems, loops, pair_types = parse_dot_bracket(dot_bracket, seq)
    bp_entropy = base_pair_entropy(seq)
    seq_entropy = sequence_entropy(seq)
    graph_stats = graph_topology_stats(seq)
    gc = gc_fraction(seq)

    mg2_proxy = 1 if loops >= 3 and stems >= 2 and mfe < -20 else 0

    for resid, base in enumerate(seq, start=1):
        record = {
            "target_id": tid,
            "conformation": 1,  # placeholder if not available
            "resid": resid,
            "resname": base,
            "dot_bracket": dot_bracket[resid - 1],
            "is_paired": int((resid - 1) in pair_map),
            "paired_with": pair_map.get(resid - 1, -1) + 1 if (resid - 1) in pair_map else -1,
            "mfe": mfe,
            "sequence_entropy": seq_entropy,
            "basepair_entropy": bp_entropy,
            "gc_content": gc,
            "avg_degree": graph_stats["avg_degree"],
            "avg_clustering": graph_stats["avg_clustering"],
            "num_stems": stems,
            "num_loops": loops,
            "mg2_affinity_proxy": mg2_proxy,
            "pair_AU": pair_types.get("AU", 0),
            "pair_GC": pair_types.get("CG", 0),
            "pair_GU": pair_types.get("GU", 0)
        }
        records.append(record)

    dotbracket_strings.append({"target_id": tid, "dot_bracket": dot_bracket})

# === Save Outputs === #
feature_df = pd.DataFrame(records)
feature_df.to_csv(OUTPUT_FEATURES, index=False)
print(f"\n Saved per-residue RNA-aware features to `{OUTPUT_FEATURES}` — shape: {feature_df.shape}")

db_df = pd.DataFrame(dotbracket_strings)
db_df.to_csv(DOTBRACKET_OUTPUT, index=False)
print(f" Dot-bracket strings saved to `{DOTBRACKET_OUTPUT}`")
