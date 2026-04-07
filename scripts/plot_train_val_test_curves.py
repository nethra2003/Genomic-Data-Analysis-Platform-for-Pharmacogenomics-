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
# CONFIG (YOUR EXACT PATHS)
# ==============================
DATA_PATH = Path(r"E:\genomics_project\outputs\training_dataset.csv")

XGB_MODEL_PATH = Path(r"E:\genomics_project\models\xgboost_model_chunked.json")  # only for reference, not used here
FT_EPOCH_PATHS = [
    Path(r"E:\genomics_project\models\ft_transformer_epoch1.pth"),
    Path(r"E:\genomics_project\models\ft_transformer_epoch2.pth"),
    Path(r"E:\genomics_project\models\ft_transformer_epoch3.pth"),
]
ENCODER_PATH = Path(r"E:\genomics_project\models\feature_encoders.pkl")
LABEL_MAP_PATH = Path(r"E:\genomics_project\models\label_map.pkl")

MODEL_DIR = Path(r"E:\genomics_project\models")
PLOTS_DIR = MODEL_DIR / "training_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ROWS = 200_000   # to avoid loading all 14M rows; set to None to use full
RANDOM_STATE = 42
XGB_NUM_ROUNDS = 80      # number of boosting rounds for XGBoost curves
FT_BATCH_SIZE = 2048

# ==============================
# FTTransformer MODEL DEFINITION
# (matches your train/eval code)
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

# categorical and numeric columns for FT
cat_cols = list(encoders.keys())
if "Response_Type" in cat_cols:
    cat_cols.remove("Response_Type")

# numeric columns we used in FT eval
num_cols = ["CHROM", "POS", "QUAL"]
n_cat = [len(encoders[c]) for c in cat_cols]

# ==============================
# LOAD & SAMPLE DATA
# ==============================
print("📂 Loading dataset sample for curves...")

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

print(f"✅ Using {len(df):,} rows for plotting curves.")

# ==============================
# TRAIN / VAL / TEST SPLIT (60/20/20)
# ==============================
temp, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Response_Type_enc"],
    random_state=RANDOM_STATE,
)
train_df, val_df = train_test_split(
    temp,
    test_size=0.25,  # 0.25 of 0.8 = 0.2
    stratify=temp["Response_Type_enc"],
    random_state=RANDOM_STATE,
)

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

y_train = train_df["Response_Type_enc"].to_numpy()
y_val = val_df["Response_Type_enc"].to_numpy()
y_test = test_df["Response_Type_enc"].to_numpy()

# ======================================================
# PART 1 — XGBOOST: TRAIN vs VAL vs TEST (ACCURACY & LOGLOSS)
# ======================================================
print("\n🚀 Generating XGBoost train/val/test curves (retraining on sample)...")

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

    # encode categoricals using your encoders
    for col, mapping in encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(np.float32)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y = df_subset["Response_Type_enc"].to_numpy()
    return xgb.DMatrix(X, label=y)

dtrain = prepare_xgb_matrix(train_df)
dval = prepare_xgb_matrix(val_df)
dtest = prepare_xgb_matrix(test_df)

params = {
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
watchlist = [(dtrain, "train"), (dval, "val"), (dtest, "test")]

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=XGB_NUM_ROUNDS,
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=False,
)

iters = list(range(1, XGB_NUM_ROUNDS + 1))
train_logloss = evals_result["train"]["mlogloss"]
val_logloss   = evals_result["val"]["mlogloss"]
test_logloss  = evals_result["test"]["mlogloss"]

train_acc = [1.0 - e for e in evals_result["train"]["merror"]]
val_acc   = [1.0 - e for e in evals_result["val"]["merror"]]
test_acc  = [1.0 - e for e in evals_result["test"]["merror"]]

# ---- Plot XGBoost Accuracy (with zoom + styling) ----
plt.figure(figsize=(8, 5))
plt.plot(iters, train_acc, marker="o", linestyle="-", label="Train Accuracy")
plt.plot(iters, val_acc,   marker="s", linestyle="--", label="Validation Accuracy")
plt.plot(iters, test_acc,  marker="^", linestyle=":", label="Test Accuracy")
plt.xlabel("Boosting Rounds")
plt.ylabel("Accuracy")
plt.title("XGBoost — Train vs Validation vs Test Accuracy")
plt.grid(True)
plt.legend()

all_acc = np.array(train_acc + val_acc + test_acc)
y_min = all_acc.min() - 0.01
y_max = all_acc.max() + 0.01
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "xgb_train_val_test_accuracy.png")

# ---- Plot XGBoost LogLoss (with zoom + styling) ----
plt.figure(figsize=(8, 5))
plt.plot(iters, train_logloss, marker="o", linestyle="-", label="Train LogLoss")
plt.plot(iters, val_logloss,   marker="s", linestyle="--", label="Validation LogLoss")
plt.plot(iters, test_logloss,  marker="^", linestyle=":", label="Test LogLoss")
plt.xlabel("Boosting Rounds")
plt.ylabel("LogLoss")
plt.title("XGBoost — Train vs Validation vs Test LogLoss")
plt.grid(True)
plt.legend()

all_ll = np.array(train_logloss + val_logloss + test_logloss)
y_min_ll = all_ll.min() - 0.01
y_max_ll = all_ll.max() + 0.01
plt.ylim(y_min_ll, y_max_ll)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "xgb_train_val_test_logloss.png")

print("✅ Saved XGBoost curves to:", PLOTS_DIR)

# ======================================================
# PART 2 — FTTRANSFORMER: TRAIN vs VAL vs TEST (EPOCHS)
# ======================================================
print("\n🚀 Generating FTTransformer train/val/test curves from saved epochs...")

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
Xc_te, Xn_te, y_te = prepare_ft_arrays(test_df)

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

epochs = []
ft_train_acc = []
ft_val_acc = []
ft_test_acc = []
ft_train_ll = []
ft_val_ll = []
ft_test_ll = []

for ck_path in FT_EPOCH_PATHS:
    if not ck_path.exists():
        print(f"⚠️ Skipping missing checkpoint: {ck_path}")
        continue

    name = ck_path.stem
    digits = "".join([c for c in name if c.isdigit()])
    epoch_num = int(digits) if digits else len(epochs) + 1

    print(f"🔍 Evaluating FT checkpoint {ck_path.name} (epoch {epoch_num})")

    model = FTTransformer(n_num=Xn_tr.shape[1], n_cat=n_cat, num_classes=num_classes)
    state = torch.load(ck_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)

    # Train
    y_pred_tr, prob_tr = batch_predict_ft(model, Xc_tr, Xn_tr)
    acc_tr = accuracy_score(y_tr, y_pred_tr)
    ll_tr = log_loss(y_tr, prob_tr, labels=list(range(num_classes)))

    # Val
    y_pred_val, prob_val = batch_predict_ft(model, Xc_val, Xn_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    ll_val = log_loss(y_val, prob_val, labels=list(range(num_classes)))

    # Test
    y_pred_te, prob_te = batch_predict_ft(model, Xc_te, Xn_te)
    acc_te = accuracy_score(y_te, y_pred_te)
    ll_te = log_loss(y_te, prob_te, labels=list(range(num_classes)))

    epochs.append(epoch_num)
    ft_train_acc.append(acc_tr)
    ft_val_acc.append(acc_val)
    ft_test_acc.append(acc_te)
    ft_train_ll.append(ll_tr)
    ft_val_ll.append(ll_val)
    ft_test_ll.append(ll_te)

    print(f"  ➜ Train: {acc_tr:.4f}, Val: {acc_val:.4f}, Test: {acc_te:.4f}")

if epochs:
    order = np.argsort(epochs)
    ep_sorted = np.array(epochs)[order]
    tr_acc = np.array(ft_train_acc)[order]
    val_acc = np.array(ft_val_acc)[order]
    te_acc = np.array(ft_test_acc)[order]
    tr_ll = np.array(ft_train_ll)[order]
    val_ll = np.array(ft_val_ll)[order]
    te_ll = np.array(ft_test_ll)[order]

    # ---- FT Accuracy (with zoom + styling) ----
    plt.figure(figsize=(8, 5))
    plt.plot(ep_sorted, tr_acc, marker="o", linestyle="-", label="Train Accuracy")
    plt.plot(ep_sorted, val_acc, marker="s", linestyle="--", label="Validation Accuracy")
    plt.plot(ep_sorted, te_acc,  marker="^", linestyle=":", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("FTTransformer — Train vs Validation vs Test Accuracy")
    plt.grid(True)
    plt.legend()

    all_ft_acc = np.concatenate([tr_acc, val_acc, te_acc])
    y_min_ft = all_ft_acc.min() - 0.01
    y_max_ft = all_ft_acc.max() + 0.01
    plt.ylim(y_min_ft, y_max_ft)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ft_train_val_test_accuracy.png")

    # ---- FT LogLoss (with zoom + styling) ----
    plt.figure(figsize=(8, 5))
    plt.plot(ep_sorted, tr_ll, marker="o", linestyle="-", label="Train LogLoss")
    plt.plot(ep_sorted, val_ll, marker="s", linestyle="--", label="Validation LogLoss")
    plt.plot(ep_sorted, te_ll,  marker="^", linestyle=":", label="Test LogLoss")
    plt.xlabel("Epoch")
    plt.ylabel("LogLoss")
    plt.title("FTTransformer — Train vs Validation vs Test LogLoss")
    plt.grid(True)
    plt.legend()

    all_ft_ll = np.concatenate([tr_ll, val_ll, te_ll])
    y_min_ft_ll = all_ft_ll.min() - 0.02
    y_max_ft_ll = all_ft_ll.max() + 0.02
    plt.ylim(y_min_ft_ll, y_max_ft_ll)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ft_train_val_test_logloss.png")

    print("✅ Saved FTTransformer curves to:", PLOTS_DIR)
else:
    print("⚠️ No valid FTTransformer checkpoints evaluated.")

print("\n🎉 Done! All plots generated in:", PLOTS_DIR)
