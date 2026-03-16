import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from Bio import SeqIO
from scipy.stats import entropy
from scipy.linalg import eigh

# === CONFIG === #
SEQ_FILE = "phase1_cleaned_sequences.csv"
MSA_PSSM_FILE = "features/msa_pssm_conservation.npz"
GRAPH_STATS_FILE = "features/graph_topology_stats.csv"
OUTPUT_SEQ_FEATURES = "features/msa_graph_features.csv"
OUTPUT_RES_FEATURES = "features/msa_graph_features_residue.csv"
GRAPH_DIR = "graphs"
DOTBRACKET_DIR = "features/dotbracket"
ALLOWED_BASES = ['A', 'C', 'G', 'U']
MAX_EIGEN = 10

# === Ensure directories exist === #
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs("features", exist_ok=True)

# === Load Sequences === #
print(" Loading cleaned sequences...")
df_seq = pd.read_csv(SEQ_FILE)
clean_seqs = dict(zip(df_seq["target_id"], df_seq["clean_sequence"]))

# === Load MSA PSSM Features === #
print(" Loading MSA PSSM data...")
msa_data = np.load(MSA_PSSM_FILE)

# === Helper: Dot-bracket base pairs === #
def parse_dotbracket(dot_bracket):
    stack, pairs = [], {}
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            pairs[i] = j
            pairs[j] = i
    return pairs

# === Helper: Compute Laplacian spectrum features === #
def compute_laplacian_features(G):
    if len(G) == 0:
        return [0.0] * MAX_EIGEN, 0.0, 0.0

    L = nx.laplacian_matrix(G).todense()
    eigenvalues = eigh(L, eigvals_only=True)
    eigenvalues = np.sort(np.real(eigenvalues))

    eigvals_padded = np.zeros(MAX_EIGEN)
    eigvals_to_use = min(MAX_EIGEN, len(eigenvalues))
    eigvals_padded[:eigvals_to_use] = eigenvalues[:eigvals_to_use]

    probs = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else np.ones_like(eigenvalues) / len(eigenvalues)
    spectral_entropy = -np.sum(probs * np.log2(probs + 1e-9))

    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0

    return eigvals_padded.tolist(), spectral_entropy, spectral_gap

# === Graph Construction === #
print("\U0001F9EC Building RNA graphs...")
graph_stats = []
for _, row in tqdm(df_seq.iterrows(), total=len(df_seq)):
    tid = row["target_id"]
    seq = row["clean_sequence"]
    G = nx.Graph()

    for i, base in enumerate(seq):
        G.add_node(i, base=base)

    for i in range(len(seq) - 1):
        G.add_edge(i, i + 1, type="backbone")

    dbn_path = os.path.join(DOTBRACKET_DIR, f"{tid}.db")
    if os.path.exists(dbn_path):
        with open(dbn_path, "r") as f:
            dot = f.read().strip()
            for i, j in parse_dotbracket(dot).items():
                if not G.has_edge(i, j):
                    G.add_edge(i, j, type="basepair")

    with open(os.path.join(GRAPH_DIR, f"{tid}.gpickle"), "wb") as f:
        pickle.dump(G, f)

    degrees = [deg for _, deg in G.degree()]
    clustering = nx.average_clustering(G) if len(G) > 1 else 0.0
    eigvals, spec_entropy, spec_gap = compute_laplacian_features(G)

    stat_row = {
        "target_id": tid,
        "avg_degree": np.mean(degrees),
        "avg_clustering": clustering,
        "num_nodes": len(G),
        "spectral_entropy": spec_entropy,
        "spectral_gap": spec_gap
    }
    for k in range(MAX_EIGEN):
        stat_row[f"laplacian_eigval_{k}"] = eigvals[k]

    graph_stats.append(stat_row)

print(f" Saved {len(graph_stats)} graphs.")

# === Convert per-sequence MSA + graph to DataFrame === #
graph_df = pd.DataFrame(graph_stats)
msa_seq_df = pd.DataFrame([
    {
        "target_id": tid,
        **{f"msa_feat_{j}": val for j, val in enumerate(mat.flatten())}
    }
    for tid, mat in msa_data.items()
])
merged_seq = graph_df.merge(msa_seq_df, on="target_id", how="inner")
merged_seq.to_csv(OUTPUT_SEQ_FEATURES, index=False)
print(f" Per-target MSA + graph features saved to `{OUTPUT_SEQ_FEATURES}` with shape {merged_seq.shape}")

# === Generate Per-Residue MSA features === #
print("\U0001F522 Generating per-residue MSA PSSM + entropy + conservation features...")
residue_rows = []
for tid, mat in msa_data.items():
    seq = clean_seqs.get(tid)
    if seq is None or mat.shape[0] != len(seq):
        continue

    for i in range(len(seq)):
        row = {
            "target_id": tid,
            "resid": i + 1
        }
        for j, base in enumerate(ALLOWED_BASES):
            row[f"msa_pssm_{base}"] = mat[i, j]
        row["msa_entropy"] = mat[i, 4] if mat.shape[1] > 4 else 0.0
        row["msa_conservation"] = mat[i, 5] if mat.shape[1] > 5 else 0.0
        residue_rows.append(row)

df_residue = pd.DataFrame(residue_rows)
df_residue.to_csv(OUTPUT_RES_FEATURES, index=False)
print(f" Per-residue MSA features saved to `{OUTPUT_RES_FEATURES}` with shape {df_residue.shape}")