import os
import torch
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm

# ======================================================
# CONFIGURATION
# ======================================================
DATA_PATH = r"E:\genomics_project\outputs\training_dataset.csv"
MODEL_DIR = Path(r"E:\genomics_project\models")
OUTPUT_PATH = Path(r"E:\genomics_project\outputs\predictions_combined.csv")
CHUNK_SIZE = 500_000      # CSV chunks
BATCH_SIZE = 5_000        # Mini-batches for FT Transformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model_chunked.json"
FT_MODEL_PATH = MODEL_DIR / "ft_transformer_epoch3.pth"
ENCODER_PATH = MODEL_DIR / "feature_encoders.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_map.pkl"

# ======================================================
# FT Transformer definition
# ======================================================
class FTTransformer(nn.Module):
    def __init__(self, n_num, n_cat, num_classes, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(size + 1, embed_dim) for size in n_cat])  # +1 for unknowns
        self.num_linear = nn.Linear(n_num, embed_dim) if n_num > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, X_cat, X_num):
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

# ======================================================
# LOAD MODELS & ENCODERS
# ======================================================
print("📦 Loading models and encoders...")

booster = xgb.Booster()
booster.load_model(str(XGBOOST_MODEL_PATH))

encoders = joblib.load(ENCODER_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}

cat_cols = list(encoders.keys())
if "Response_Type" in cat_cols:
    cat_cols.remove("Response_Type")

n_cat = [len(encoders[c]) for c in cat_cols]
n_num = 3  # CHROM, POS, QUAL
num_classes = len(label_map)

print(f"🧩 Using {len(cat_cols)} categorical columns for FT Transformer:")
print(cat_cols)

ft_model = FTTransformer(n_num=n_num, n_cat=n_cat, num_classes=num_classes)
state_dict = torch.load(FT_MODEL_PATH, map_location=DEVICE)
ft_model.load_state_dict(state_dict, strict=False)
ft_model.to(DEVICE)
ft_model.eval()

expected_features = [
    "CHROM", "POS", "ID", "REF", "ALT", "QUAL", "Gene", "Consequence",
    "IMPACT", "Variant", "Drug", "Condition", "Dosage", "Recommendation"
]

# ======================================================
# PREDICTION LOOP
# ======================================================
print(f"📂 Streaming dataset from {DATA_PATH} ...")
predictions = []

reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)
for i, chunk in enumerate(reader):
    print(f"🧩 Processing chunk {i+1}")
    chunk = chunk.replace({".": np.nan})
    chunk = chunk.drop(columns=["Response_Type"], errors="ignore")

    # ----- Prepare XGBoost Input -----
    X_xgb = chunk[[f for f in expected_features if f in chunk.columns]].copy()
    if "ID" not in X_xgb.columns:
        X_xgb["ID"] = -1

    X_xgb = X_xgb.replace({".": -1}).fillna(-1)
    for col in X_xgb.columns:
        if X_xgb[col].dtype == "object":
            if col in encoders:
                mapping = encoders[col]
                X_xgb[col] = X_xgb[col].astype(str).map(mapping).fillna(-1).astype(np.float32)
            else:
                X_xgb[col] = X_xgb[col].astype(str).factorize()[0].astype(np.float32)
        else:
            X_xgb[col] = X_xgb[col].astype(np.float32)

    dtest = xgb.DMatrix(X_xgb, enable_categorical=False)
    y_pred_probs_xgb = booster.predict(dtest)

    # ----- Prepare FT Transformer Input -----
    for col in cat_cols:
        chunk.loc[:, col] = chunk[col].astype(str).map(encoders[col]).fillna(-1).astype(np.int64)

    X_cat_all = chunk[cat_cols].to_numpy(np.int64)
    X_num_all = chunk[["CHROM", "POS", "QUAL"]].to_numpy(np.float32)

    preds_ft_all = []

    # Predict in mini-batches
    for start in tqdm(range(0, len(chunk), BATCH_SIZE), desc=f"  🔹 FT batches ({i+1})"):
        end = start + BATCH_SIZE
        X_cat = torch.tensor(X_cat_all[start:end], dtype=torch.long).to(DEVICE)
        X_num = torch.tensor(X_num_all[start:end], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            outputs = ft_model(X_cat, X_num)
            batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds_ft_all.append(batch_probs)
        del X_cat, X_num, outputs, batch_probs
        torch.cuda.empty_cache()

    y_pred_probs_ft = np.concatenate(preds_ft_all, axis=0)

    # ----- Combine Predictions -----
    min_len = min(len(y_pred_probs_xgb), len(y_pred_probs_ft))
    combined_probs = (y_pred_probs_xgb[:min_len] + y_pred_probs_ft[:min_len]) / 2
    y_pred_combined = np.argmax(combined_probs, axis=1)

    # Map to labels
    labels = [inv_label_map.get(int(lbl), "Unknown") for lbl in y_pred_combined]

    # Save result
    chunk_result = pd.DataFrame({
        "CHROM": chunk["CHROM"].iloc[:min_len],
        "POS": chunk["POS"].iloc[:min_len],
        "Gene": chunk["Gene"].iloc[:min_len],
        "Predicted_Response": labels
    })
    predictions.append(chunk_result)

    del chunk, X_xgb, y_pred_probs_xgb, y_pred_probs_ft
    torch.cuda.empty_cache()

# Save all predictions
final_df = pd.concat(predictions, ignore_index=True)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"🎯 Combined predictions saved to: {OUTPUT_PATH}")
