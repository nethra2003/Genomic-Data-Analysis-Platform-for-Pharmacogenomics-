import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

# =============================
# CONFIGURATION
# =============================
DATA_PATH = r"E:\genomics_project\outputs\training_dataset.csv"
MODEL_DIR = Path(r"E:\genomics_project\models")
FT_DATA_DIR = Path(r"E:\genomics_project\ft_data\chunks")

MODEL_PATH = MODEL_DIR / "ft_transformer_epoch3.pth"   # adjust if you saved a different epoch
ENCODER_PATH = MODEL_DIR / "feature_encoders.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_map.pkl"

CHUNK_SIZE = 200_000
BATCH_SIZE = 512  # reduce to 256 if still memory issues

# =============================
# MODEL DEFINITION
# =============================
class FTTransformer(nn.Module):
    def __init__(self, n_num, n_cat, num_classes, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(size + 1, embed_dim) for size in n_cat])
        self.num_linear = nn.Linear(n_num, embed_dim) if n_num > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, X_cat, X_num):
        # clamp categorical indices
        for i, emb in enumerate(self.embeds):
            X_cat[:, i] = torch.clamp(X_cat[:, i], 0, emb.num_embeddings - 1)

        cat_embeds = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeds)]
        cat_embeds = torch.stack(cat_embeds, dim=1)

        if X_num is not None and self.num_linear is not None:
            num_embed = self.num_linear(X_num).unsqueeze(1)
            cat_embeds = torch.cat([cat_embeds, num_embed], dim=1)

        encoded = self.encoder(cat_embeds)
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)

# =============================
# EVALUATION
# =============================
print("📦 Loading FT Transformer model and encoders...")
encoders = joblib.load(ENCODER_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}

n_cat = [len(v) for v in encoders.values()]
n_num = 3
num_classes = len(label_map)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FTTransformer(n_num, n_cat, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("📂 Streaming dataset for evaluation...")
reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)

y_true_all, y_pred_all = [], []

for i, chunk in enumerate(reader):
    print(f"🧩 Processing chunk {i+1}")

    if "Response_Type" not in chunk.columns:
        print("⚠️ Skipping chunk without Response_Type.")
        continue

    chunk = chunk.dropna(subset=["Response_Type"])
    y_true = chunk["Response_Type"].map(label_map).dropna()
    chunk = chunk.loc[y_true.index]
    y_true = y_true.astype(int)

    # Separate numeric and categorical
    num_cols = ["CHROM", "POS", "QUAL"]
    cat_cols = [c for c in encoders.keys() if c in chunk.columns]

    X_num = chunk[num_cols].to_numpy(np.float32)
    X_cat = np.zeros((len(chunk), len(cat_cols)), dtype=np.int32)
    for j, col in enumerate(cat_cols):
        chunk[col] = chunk[col].astype(str)
        chunk[col] = chunk[col].map(encoders[col]).fillna(-1).astype(np.int32)
        X_cat[:, j] = chunk[col].values

    X_cat_t = torch.tensor(X_cat, dtype=torch.long, device=device)
    X_num_t = torch.tensor(X_num, dtype=torch.float32, device=device)

    # Mini-batch inference to prevent OOM
    preds_all = []
    with torch.no_grad():
        for start in range(0, len(X_cat_t), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_cat = X_cat_t[start:end]
            batch_num = X_num_t[start:end]
            outputs = model(batch_cat, batch_num)
            preds = outputs.argmax(dim=1).cpu().numpy()
            preds_all.append(preds)

    y_pred = np.concatenate(preds_all)
    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)

print("✅ Evaluation completed for all chunks.")
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# =============================
# METRICS
# =============================
acc = accuracy_score(y_true_all, y_pred_all)
prec = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
rec = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)

print("\n📊 Evaluation Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=[inv_label_map[i] for i in sorted(inv_label_map.keys())]))

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix — FT Transformer")
plt.colorbar()
tick_marks = np.arange(len(inv_label_map))
plt.xticks(tick_marks, [inv_label_map[i] for i in sorted(inv_label_map.keys())], rotation=45, ha="right")
plt.yticks(tick_marks, [inv_label_map[i] for i in sorted(inv_label_map.keys())])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(MODEL_DIR / "ft_transformer_confusion_matrix.png")
plt.show()
print(f"📈 Confusion matrix saved to {MODEL_DIR / 'ft_transformer_confusion_matrix.png'}")
