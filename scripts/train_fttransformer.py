import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# =============================
# CONFIGURATION
# =============================
DATA_PATH = r"E:\genomics_project\outputs\training_dataset.csv"
CHUNK_SIZE = 200_000
FT_DATA_DIR = Path(r"E:\genomics_project\ft_data\chunks")
MODEL_DIR = Path(r"E:\genomics_project\models")
FT_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_PATH = MODEL_DIR / "feature_encoders.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_map.pkl"

# =============================
# STEP 1 — PREPROCESSING
# =============================
def preprocess_and_chunk():
    print(f"📂 Loading structure from {DATA_PATH} ...")
    first_chunk = pd.read_csv(DATA_PATH, nrows=20)
    cat_cols = first_chunk.select_dtypes(include="object").columns.tolist()
    num_cols = first_chunk.select_dtypes(exclude="object").columns.tolist()
    drop_cols = ["ID"]
    cat_cols = [c for c in cat_cols if c not in drop_cols]
    num_cols = [c for c in num_cols if c not in drop_cols]

    print(f"🔤 Categorical columns: {cat_cols}")
    print(f"🔢 Numeric columns: {num_cols}")

    encoders = {}
    print("🔍 Scanning dataset to fit categorical encoders...")
    reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)
    for i, chunk in enumerate(reader):
        for col in cat_cols:
            chunk[col] = chunk[col].astype(str)
            if col not in encoders:
                encoders[col] = set()
            encoders[col].update(chunk[col].unique())
        print(f"  ➜ Scanned chunk {i+1}")

    # Create mappings
    for col in encoders:
        encoders[col] = {val: idx for idx, val in enumerate(sorted(encoders[col]))}

    # Build label map
    reader = pd.read_csv(DATA_PATH, chunksize=1_000_000, usecols=["Response_Type"])
    responses = set()
    for chunk in reader:
        responses.update(chunk["Response_Type"].dropna().unique())
    label_map = {val: i for i, val in enumerate(sorted(responses))}

    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(label_map, LABEL_MAP_PATH)
    print(f"💾 Saved encoders and label map in {MODEL_DIR}")

    print("🚀 Converting CSV to memory-efficient chunks...")
    reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)
    for i, chunk in enumerate(reader):
        chunk = chunk.drop(columns=["ID"], errors="ignore")
        chunk = chunk.dropna(subset=["Response_Type"])

        for col in cat_cols:
            chunk[col] = chunk[col].astype(str).map(encoders[col])
            chunk[col] = chunk[col].fillna(-1).astype(int)
        chunk["Response_Type"] = chunk["Response_Type"].map(label_map)

        X_cat = chunk[cat_cols].to_numpy(np.int32)
        X_num = chunk[num_cols].to_numpy(np.float32) if len(num_cols) > 0 else np.zeros((len(chunk), 0), dtype=np.float32)
        y = chunk["Response_Type"].to_numpy(np.int32)

        np.savez_compressed(FT_DATA_DIR / f"chunk_{i:04d}.npz", X_cat=X_cat, X_num=X_num, y=y)
        print(f"✅ Saved chunk_{i:04d}.npz ({len(chunk)} rows)")

    print("🎉 Preprocessing complete! All chunks saved.")


# =============================
# STEP 2 — FT Transformer Model
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
        # Clamp categorical indices
        for i, emb in enumerate(self.embeds):
            X_cat[:, i] = torch.clamp(X_cat[:, i], 0, emb.num_embeddings - 1)

        cat_embeds = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeds)]
        cat_embeds = torch.stack(cat_embeds, dim=1)
        if X_num is not None and self.num_linear is not None and X_num.shape[1] > 0:
            num_embed = self.num_linear(X_num).unsqueeze(1)
            cat_embeds = torch.cat([cat_embeds, num_embed], dim=1)
        encoded = self.encoder(cat_embeds)
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)


# =============================
# STEP 3 — TRAINING LOOP
# =============================
def train_ft_transformer():
    if not list(FT_DATA_DIR.glob("chunk_*.npz")):
        preprocess_and_chunk()

    encoders = joblib.load(ENCODER_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)
    num_classes = len(label_map)
    n_cat = [len(v) for v in encoders.values()]

    # Automatically infer number of numeric columns from any chunk
    first_chunk_file = sorted(FT_DATA_DIR.glob("chunk_*.npz"))[0]
    sample = np.load(first_chunk_file)
    n_num = sample["X_num"].shape[1]
    sample.close()

    print(f"📊 Found {len(n_cat)} categorical columns, {n_num} numeric, {num_classes} classes.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FTTransformer(n_num=n_num, n_cat=n_cat, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 3
    BATCH_SIZE = 1024
    chunk_files = sorted(FT_DATA_DIR.glob("chunk_*.npz"))

    for epoch in range(EPOCHS):
        print(f"\n🌍 Epoch {epoch+1}/{EPOCHS}")
        for chunk_file in chunk_files:
            data = np.load(chunk_file)
            X_cat = torch.tensor(data["X_cat"], dtype=torch.long).to(device)
            X_num = torch.tensor(data["X_num"], dtype=torch.float32).to(device)
            y = torch.tensor(data["y"], dtype=torch.long).to(device)
            data.close()

            # Filter invalid labels
            valid_mask = (y >= 0) & (y < num_classes)
            if valid_mask.sum() == 0:
                continue

            X_cat = X_cat[valid_mask]
            X_num = X_num[valid_mask]
            y = y[valid_mask]

            loader = DataLoader(TensorDataset(X_cat, X_num, y), batch_size=BATCH_SIZE, shuffle=True)

            total_loss = 0
            for Xc, Xn, labels in loader:
                optimizer.zero_grad()
                preds = model(Xc, Xn)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"✅ {chunk_file.name} | avg_loss={total_loss/len(loader):.4f}")

        torch.save(model.state_dict(), MODEL_DIR / f"ft_transformer_epoch{epoch+1}.pth")
        print(f"💾 Saved checkpoint: ft_transformer_epoch{epoch+1}.pth")

    print("🎉 Training complete! Final model saved.")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    train_ft_transformer()
