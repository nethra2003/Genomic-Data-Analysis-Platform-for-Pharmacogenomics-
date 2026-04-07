import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# ==============================
# CONFIG (YOUR PATHS)
# ==============================
DATA_PATH = Path(r"E:\genomics_project\outputs\training_dataset.csv")
FT_CHECKPOINT_PATH = Path(r"E:\genomics_project\models\ft_transformer_epoch3.pth")
ENCODER_PATH = Path(r"E:\genomics_project\models\feature_encoders.pkl")
LABEL_MAP_PATH = Path(r"E:\genomics_project\models\label_map.pkl")

MODEL_DIR = Path(r"E:\genomics_project\models")
PLOTS_DIR = MODEL_DIR / "training_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ROWS = 200_000    # sample from 14M rows
RANDOM_STATE = 42
XGB_NUM_ROUNDS = 80
FT_BATCH_SIZE = 2048

# ==============================
# FTTransformer MODEL DEFINITION
# ==============================
class FTTransformer(nn.Module):
    def __init__(self, n_num, n_cat, num_classes,
                 embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(size + 1, embed_dim) for size in n_cat]
        )
        self.num_linear = nn.Linear(n_num, embed_dim) if n_num > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, X_cat, X_num):
        cat_embs = []
        for i, emb in enumerate(self.embeds):
            idx = X_cat[:, i].clamp(0, emb.num_embeddings - 1)
            cat_embs.append(emb(idx))
        cat_embeds = torch.stack(cat_embs, dim=1)
        if X_num is not None and self.num_linear is not None and X_num.shape[1] > 0:
            num_embed = self.num_linear(X_num).unsqueeze(1)
            cat_embeds = torch.cat([cat_embeds, num_embed], dim=1)
        encoded = self.encoder(cat_embeds)
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)

# ==============================
# LOAD ENCODERS + LABEL MAP
# ==============================
encoders = joblib.load(ENCODER_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# FT categorical + numeric columns
cat_cols = list(encoders.keys())
if "Response_Type" in cat_cols:
    cat_cols.remove("Response_Type")

num_cols = ["CHROM", "POS", "QUAL"]
n_cat = [len(encoders[c]) for c in cat_cols]

# ==============================
# LOAD & SAMPLE DATA
# ==============================
print("📂 Loading dataset sample for analysis...")

if SAMPLE_ROWS is None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
else:
    reader = pd.read_csv(DATA_PATH, chunksize=100_000, low_memory=False)
    parts = []
    collected = 0
    for chunk in reader:
        need = SAMPLE_ROWS - collected
        if need <= 0:
            break
        take = min(len(chunk), need)
        if len(chunk) > take:
            parts.append(chunk.sample(n=take, random_state=RANDOM_STATE))
        else:
            parts.append(chunk)
        collected += take
    df = pd.concat(parts, ignore_index=True)

df = df.dropna(subset=["Response_Type"]).reset_index(drop=True)
df["Response_Type_enc"] = df["Response_Type"].map(label_map)
df = df[df["Response_Type_enc"].notna()].copy()
df["Response_Type_enc"] = df["Response_Type_enc"].astype(int)

print(f"✅ Using {len(df):,} rows for Train/Validation + analysis.")

# ==============================
# TRAIN / VALIDATION SPLIT (80/20)
# ==============================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Response_Type_enc"],
    random_state=RANDOM_STATE,
)

print(f"Train: {len(train_df):,}, Validation: {len(val_df):,}")

y_train = train_df["Response_Type_enc"].to_numpy()
y_val = val_df["Response_Type_enc"].to_numpy()

# ======================================================
# PART 1 — XGBOOST: TRAIN vs VAL CURVES (ACCURACY & LOGLOSS)
# ======================================================
print("\n🚀 Training XGBoost on sample for Train/Val curves...")

expected_features = [
    "CHROM", "POS", "ID", "REF", "ALT", "QUAL",
    "Gene", "Consequence", "IMPACT", "Variant",
    "Drug", "Condition", "Dosage", "Recommendation",
]

def prepare_xgb_matrix(df_subset):
    X = df_subset.copy()
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features].copy()

    for col, mapping in encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(np.float32)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y = df_subset["Response_Type_enc"].to_numpy()
    return xgb.DMatrix(X, label=y)

dtrain = prepare_xgb_matrix(train_df)
dval = prepare_xgb_matrix(val_df)

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": ["mlogloss", "merror"],
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}

evals_result = {}
watchlist = [(dtrain, "train"), (dval, "val")]

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=XGB_NUM_ROUNDS,
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=False,
)

iters = list(range(1, XGB_NUM_ROUNDS + 1))
train_logloss = evals_result["train"]["mlogloss"]
val_logloss   = evals_result["val"]["mlogloss"]

train_acc = [1.0 - e for e in evals_result["train"]["merror"]]
val_acc   = [1.0 - e for e in evals_result["val"]["merror"]]

# ---- XGBoost Accuracy Curve (Train vs Val) ----
plt.figure(figsize=(8, 5))
plt.plot(iters, train_acc, marker="o", linestyle="-", label="Train Accuracy")
plt.plot(iters, val_acc,   marker="s", linestyle="--", label="Validation Accuracy")
plt.xlabel("Boosting Rounds")
plt.ylabel("Accuracy")
plt.title("XGBoost — Train vs Validation Accuracy")
plt.grid(True)
plt.legend()

all_acc = np.array(train_acc + val_acc)
plt.ylim(all_acc.min() - 0.01, all_acc.max() + 0.01)

plt.tight_layout()
# filename kept as requested
plt.savefig(PLOTS_DIR / "xgb_train_val_test_accuracy.png")

# ---- XGBoost LogLoss Curve (Train vs Val) ----
plt.figure(figsize=(8, 5))
plt.plot(iters, train_logloss, marker="o", linestyle="-", label="Train LogLoss")
plt.plot(iters, val_logloss,   marker="s", linestyle="--", label="Validation LogLoss")
plt.xlabel("Boosting Rounds")
plt.ylabel("LogLoss")
plt.title("XGBoost — Train vs Validation LogLoss")
plt.grid(True)
plt.legend()

all_ll = np.array(train_logloss + val_logloss)
plt.ylim(all_ll.min() - 0.01, all_ll.max() + 0.01)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "xgb_train_val_test_logloss.png")

print("✅ Saved XGBoost Train/Val curves.")

# ======================================================
# PART 2 — FTTRANSFORMER (SINGLE EPOCH): TRAIN vs VAL
# ======================================================
print("\n🚀 Evaluating FTTransformer (epoch3) on Train/Val...")

device = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_ft_arrays(df_subset):
    Xn = df_subset[num_cols].fillna(0).astype(np.float32).to_numpy()
    Xc = np.zeros((len(df_subset), len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        Xc[:, i] = df_subset[col].astype(str).map(encoders[col]).fillna(-1).astype(np.int64).values
    y = df_subset["Response_Type_enc"].to_numpy()
    return Xc, Xn, y

Xc_tr, Xn_tr, y_tr = prepare_ft_arrays(train_df)
Xc_val, Xn_val, y_val = prepare_ft_arrays(val_df)

def batch_predict_ft(model, Xc, Xn):
    preds = []
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xc), FT_BATCH_SIZE):
            xc = torch.tensor(Xc[i:i+FT_BATCH_SIZE], dtype=torch.long, device=device)
            xn = torch.tensor(Xn[i:i+FT_BATCH_SIZE], dtype=torch.float32, device=device)
            out = model(xc, xn)
            p = torch.softmax(out, dim=1).cpu().numpy()
            preds.append(np.argmax(p, axis=1))
            probs.append(p)
    return np.concatenate(preds), np.vstack(probs)

if FT_CHECKPOINT_PATH.exists():
    ft_model = FTTransformer(n_num=Xn_tr.shape[1], n_cat=n_cat, num_classes=num_classes)
    state = torch.load(FT_CHECKPOINT_PATH, map_location=device)
    ft_model.load_state_dict(state, strict=False)
    ft_model.to(device)

    # Train / Val evaluation
    ft_pred_tr, ft_prob_tr = batch_predict_ft(ft_model, Xc_tr, Xn_tr)
    ft_pred_val, ft_prob_val = batch_predict_ft(ft_model, Xc_val, Xn_val)

    ft_acc_tr = accuracy_score(y_tr, ft_pred_tr)
    ft_acc_val = accuracy_score(y_val, ft_pred_val)

    ft_ll_tr = log_loss(y_tr, ft_prob_tr, labels=list(range(num_classes)))
    ft_ll_val = log_loss(y_val, ft_prob_val, labels=list(range(num_classes)))

    print(f"FTTransformer (epoch3) -> Train Acc: {ft_acc_tr:.4f}, Val Acc: {ft_acc_val:.4f}")

    # ---- FT Accuracy Bar Plot (Train vs Val) ----
    plt.figure(figsize=(6, 5))
    labels_split = ["Train", "Validation"]
    acc_values = [ft_acc_tr, ft_acc_val]
    x_pos = np.arange(len(labels_split))
    plt.bar(x_pos, acc_values)
    plt.xticks(x_pos, labels_split)
    plt.ylabel("Accuracy")
    plt.title("FTTransformer (Epoch 3) — Train vs Validation Accuracy")

    y_min_ft = min(acc_values) - 0.02
    y_max_ft = max(acc_values) + 0.02
    plt.ylim(y_min_ft, y_max_ft)

    for i, v in enumerate(acc_values):
        plt.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    # filename kept as requested
    plt.savefig(PLOTS_DIR / "ft_train_val_test_accuracy.png")

    # ---- FT LogLoss Bar Plot (Train vs Val) ----
    plt.figure(figsize=(6, 5))
    ll_values = [ft_ll_tr, ft_ll_val]
    plt.bar(x_pos, ll_values)
    plt.xticks(x_pos, labels_split)
    plt.ylabel("LogLoss")
    plt.title("FTTransformer (Epoch 3) — Train vs Validation LogLoss")

    y_min_ft_ll = min(ll_values) - 0.02
    y_max_ft_ll = max(ll_values) + 0.02
    plt.ylim(y_min_ft_ll, y_max_ft_ll)

    for i, v in enumerate(ll_values):
        plt.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ft_train_val_test_logloss.png")

    print("✅ Saved FTTransformer Train/Val graphs.")
else:
    print(f"⚠️ FT checkpoint not found at: {FT_CHECKPOINT_PATH}")
    ft_acc_val = np.nan
    ft_ll_val = np.nan

# ======================================================
# PART 3 — MODEL COMPARISON: XGB vs FT ON VALIDATION
# ======================================================
print("\n📊 Comparing XGBoost vs FTTransformer on VALIDATION...")

# final validation acc from XGBoost curve (last round)
xgb_val_final_acc = val_acc[-1]
ft_val_final_acc = float(ft_acc_val) if 'ft_acc_val' in locals() else np.nan

# Accuracy comparison
plt.figure(figsize=(6, 5))
models = ["XGBoost", "FTTransformer"]
acc_values = [xgb_val_final_acc, ft_val_final_acc]
x_pos = np.arange(len(models))
plt.bar(x_pos, acc_values)
plt.xticks(x_pos, models)
plt.ylabel("Validation Accuracy")
plt.title("Model Comparison — Validation Accuracy (XGB vs FT)")

y_min_cmp = min(acc_values) - 0.02
y_max_cmp = max(acc_values) + 0.02
plt.ylim(y_min_cmp, y_max_cmp)
for i, v in enumerate(acc_values):
    plt.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_val_accuracy_comparison.png")

# LogLoss comparison
xgb_val_final_ll = val_logloss[-1]
ft_val_final_ll = float(ft_ll_val) if 'ft_ll_val' in locals() else np.nan

plt.figure(figsize=(6, 5))
ll_values = [xgb_val_final_ll, ft_val_final_ll]
plt.bar(x_pos, ll_values)
plt.xticks(x_pos, models)
plt.ylabel("Validation LogLoss")
plt.title("Model Comparison — Validation LogLoss (XGB vs FT)")

y_min_cmp_ll = min(ll_values) - 0.02
y_max_cmp_ll = max(ll_values) + 0.02
plt.ylim(y_min_cmp_ll, y_max_cmp_ll)
for i, v in enumerate(ll_values):
    plt.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_val_logloss_comparison.png")

# Split size visualization (Train vs Val)
plt.figure(figsize=(5, 4))
sizes = [len(train_df), len(val_df)]
labels = ["Train", "Validation"]
x_pos = np.arange(len(labels))
plt.bar(x_pos, sizes)
plt.xticks(x_pos, labels)
plt.ylabel("Number of Samples")
plt.title("Train vs Validation Split Size")
for i, v in enumerate(sizes):
    plt.text(i, v + max(sizes) * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "split_size_bar.png")

# ======================================================
# PART 4 — DATA / PHARMACOGENOMIC GRAPHS
# ======================================================
print("\n📊 Generating key data distribution / pharmacogenomic graphs...")

# ---- Bar Chart: Response_Type Distribution ----
resp_counts = df["Response_Type"].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(resp_counts.index, resp_counts.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Count")
plt.title("Response_Type Distribution")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "response_type_distribution.png")

# ---- Bar Chart: Top 15 Genes ----
if "Gene" in df.columns:
    gene_counts = df["Gene"].value_counts().head(15)
    plt.figure(figsize=(10, 5))
    plt.bar(gene_counts.index, gene_counts.values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Variant Count")
    plt.title("Top 15 Genes by Variant Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_genes_bar.png")

# ---- Heatmap: Gene vs Response_Type (Top 10 Genes) ----
if "Gene" in df.columns:
    top_genes = df["Gene"].value_counts().head(10).index.tolist()
    subset = df[df["Gene"].isin(top_genes)]
    pivot = pd.crosstab(subset["Gene"], subset["Response_Type"])
    pivot = pivot.loc[top_genes]

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, interpolation="nearest", cmap="viridis", aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("Response_Type")
    plt.ylabel("Gene")
    plt.title("Gene vs Response_Type (Top 10 Genes)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gene_vs_response_heatmap.png")

# ---- Chromosome Distribution ----
if "CHROM" in df.columns:
    chrom_counts = df["CHROM"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(chrom_counts.index.astype(str), chrom_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Chromosome Distribution (CHROM)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "chrom_distribution.png")

# ---- Top 15 Drugs ----
if "Drug" in df.columns:
    drug_counts = df["Drug"].value_counts().head(15)
    plt.figure(figsize=(10, 5))
    plt.bar(drug_counts.index, drug_counts.values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Count")
    plt.title("Top 15 Drugs by Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_drugs_bar.png")

print("\n🎉 All useful graphs generated in:", PLOTS_DIR)
