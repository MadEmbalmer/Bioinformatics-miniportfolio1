import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import optuna
from transformers import BertModel, BertConfig
from sklearn.metrics import f1_score

# === CONFIG === #
DATA_FILE = "features/phase2_2/rna_aware_features.csv"
MODEL_SAVE_PATH = "rnafold_surrogate_transformer.pt"
MAX_SEQ_LEN = 512 
BATCH_SIZE = 32
EPOCHS = 15
VALID_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)

# === ENCODING === #
BASE2IDX = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
PAD_IDX = 0
STRUCTURE_FEATURES = [
 'mfe', 'gc_content', 'sequence_entropy', 'basepair_entropy',
 'avg_degree', 'avg_clustering', 'num_stems', 'num_loops',
 'mg2_affinity_proxy', 'prev_is_paired', 'next_is_paired'
]

def encode_sequence(seq):
 return [BASE2IDX.get(base, 0) for base in seq[:MAX_SEQ_LEN]]

def pad_sequence(seq, max_len):
 return seq + [PAD_IDX] * (max_len - len(seq))

def pad_2d_features(matrix, max_len, feat_dim):
 padded = matrix[:max_len]
 padded += [[0.0] * feat_dim] * (max_len - len(padded))
 return padded

# === DATASET === #
class RNAfoldSurrogateDataset(Dataset):
 def __init__(self, csv_path):
 df = pd.read_csv(csv_path)
 self.inputs = []
 self.features = []
 self.labels = []
 self.distances = []

 grouped = df.groupby("target_id")
 for tid, group in grouped:
 if len(group) > MAX_SEQ_LEN:
 continue
 seq = group["resname"].tolist()
 is_paired = group["is_paired"].tolist()
 pair_with = group["paired_with"].tolist()

 enc = pad_sequence(encode_sequence(seq), MAX_SEQ_LEN)
 residue_feats = [[group[f].values[i] for f in STRUCTURE_FEATURES] for i in range(len(seq))]
 struct_feats = pad_2d_features(residue_feats, MAX_SEQ_LEN, len(STRUCTURE_FEATURES))
 label = pad_sequence(is_paired, MAX_SEQ_LEN)
 dist = pad_sequence([abs(i - j) if j > 0 else 0 for i, j in enumerate(pair_with)], MAX_SEQ_LEN)

 self.inputs.append(torch.tensor(enc, dtype=torch.long))
 self.features.append(torch.tensor(struct_feats, dtype=torch.float))
 self.labels.append(torch.tensor(label, dtype=torch.float))
 self.distances.append(torch.tensor(dist, dtype=torch.float))

 def __len__(self):
 return len(self.inputs)

 def __getitem__(self, idx):
 return self.inputs[idx], self.features[idx], self.labels[idx], self.distances[idx]

# === MODEL === #
class RNAfoldTransformer(nn.Module):
 def __init__(self, vocab_size, hidden_size, num_layers, intermediate_size, dropout, extra_feat_dim):
 super().__init__()
 config = BertConfig(
 vocab_size=vocab_size,
 hidden_size=hidden_size,
 num_hidden_layers=num_layers,
 num_attention_heads=4,
 intermediate_size=intermediate_size,
 hidden_dropout_prob=dropout,
 attention_probs_dropout_prob=dropout,
 max_position_embeddings=MAX_SEQ_LEN,
 pad_token_id=PAD_IDX
 )
 self.base_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_IDX)
 self.feature_proj = nn.Linear(extra_feat_dim, hidden_size)
 self.transformer = BertModel(config)
 self.classifier = nn.Linear(hidden_size, 1)
 self.regressor = nn.Linear(hidden_size, 1)

 def forward(self, tokens, features):
 mask = tokens != PAD_IDX
 token_embed = self.base_embedding(tokens)
 feat_embed = self.feature_proj(features)
 embed = token_embed + feat_embed
 out = self.transformer(inputs_embeds=embed, attention_mask=mask).last_hidden_state
 logits = self.classifier(out).squeeze(-1)
 distances = self.regressor(out).squeeze(-1)
 return logits, distances

# === HYPERPARAMETER SEARCH === #
def objective(trial):
 hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
 num_layers = trial.suggest_int("num_layers", 2, 4)
 intermediate_size = trial.suggest_categorical("intermediate_size", [128, 256, 512])
 dropout = trial.suggest_float("dropout", 0.1, 0.3)
 lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

 dataset = RNAfoldSurrogateDataset(DATA_FILE)
 val_size = int(VALID_SPLIT * len(dataset))
 train_size = len(dataset) - val_size
 train_set, val_set = random_split(dataset, [train_size, val_size])
 train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
 val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

 model = RNAfoldTransformer(
 vocab_size=5, hidden_size=hidden_size, num_layers=num_layers,
 intermediate_size=intermediate_size, dropout=dropout,
 extra_feat_dim=len(STRUCTURE_FEATURES)
 ).to(DEVICE)

 cls_loss_fn = nn.BCEWithLogitsLoss()
 reg_loss_fn = nn.MSELoss()
 optimizer = optim.AdamW(model.parameters(), lr=lr)

 for _ in range(EPOCHS):
 model.train()
 for tokens, feats, labels, dists in train_loader:
 tokens, feats, labels, dists = tokens.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE), dists.to(DEVICE)
 optimizer.zero_grad()
 logits, pred_dists = model(tokens, feats)
 loss_cls = cls_loss_fn(logits, labels)
 loss_dist = reg_loss_fn(pred_dists, dists)
 loss = loss_cls + 0.5 * loss_dist
 loss.backward()
 optimizer.step()

 model.eval()
 all_preds, all_labels = [], []
 with torch.no_grad():
 for tokens, feats, labels, _ in val_loader:
 tokens, feats, labels = tokens.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
 logits, _ = model(tokens, feats)
 preds = torch.sigmoid(logits) > 0.5
 all_preds.extend(preds.cpu().numpy().flatten())
 all_labels.extend(labels.cpu().numpy().flatten())

 return f1_score(all_labels, all_preds)

# === RUN STUDY === #
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_trial.params
print(" Best Hyperparameters:", best_params)

# === FINAL TRAINING === #
print(" Training final model on full dataset...")
final_model = RNAfoldTransformer(
 vocab_size=5,
 hidden_size=best_params["hidden_size"],
 num_layers=best_params["num_layers"],
 intermediate_size=best_params["intermediate_size"],
 dropout=best_params["dropout"],
 extra_feat_dim=len(STRUCTURE_FEATURES)
).to(DEVICE)

dataset = RNAfoldSurrogateDataset(DATA_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
cls_loss_fn = nn.BCEWithLogitsLoss()
reg_loss_fn = nn.MSELoss()
optimizer = optim.AdamW(final_model.parameters(), lr=best_params["lr"])

for epoch in range(1, EPOCHS + 1):
 final_model.train()
 epoch_loss = 0.0
 for tokens, feats, labels, dists in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
 tokens, feats, labels, dists = tokens.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE), dists.to(DEVICE)
 optimizer.zero_grad()
 logits, pred_dists = final_model(tokens, feats)
 loss_cls = cls_loss_fn(logits, labels)
 loss_dist = reg_loss_fn(pred_dists, dists)
 loss = loss_cls + 0.5 * loss_dist
 loss.backward()
 optimizer.step()
 epoch_loss += loss.item()
 print(f" Epoch {epoch}: Loss = {epoch_loss:.4f}")

torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
print(f" Final model saved to {MODEL_SAVE_PATH}")
