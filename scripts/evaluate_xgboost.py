import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
DATA_PATH = r"E:\genomics_project\outputs\training_dataset.csv"
MODEL_PATH = r"E:\genomics_project\models\xgboost_model_chunked.json"
ENCODER_PATH = r"E:\genomics_project\models\feature_encoders.pkl"
LABEL_MAP_PATH = r"E:\genomics_project\models\label_map.pkl"
CHUNK_SIZE = 200_000

# ==============================
# LOAD MODEL + ENCODERS
# ==============================
print("📦 Loading model and encoders...")
booster = xgb.Booster()
booster.load_model(MODEL_PATH)

encoders = joblib.load(ENCODER_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}

# Include ID since it was in training
expected_features = [
    'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL',
    'Gene', 'Consequence', 'IMPACT', 'Variant',
    'Drug', 'Condition', 'Dosage', 'Recommendation'
]

print("📂 Streaming dataset for evaluation...")

y_true_all, y_pred_all = [], []

reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)

for i, chunk in enumerate(reader):
    print(f"🧩 Processing chunk {i+1}")

    if "Response_Type" not in chunk.columns:
        print("❌ Skipping chunk — missing Response_Type.")
        continue

    chunk = chunk.dropna(subset=["Response_Type"]).copy()
    if chunk.empty:
        continue

    # Map labels
    chunk["Response_Type"] = chunk["Response_Type"].map(label_map)
    chunk = chunk[chunk["Response_Type"].notna()]
    y_true = chunk["Response_Type"].astype(int)

    # Add dummy ID if missing
    if "ID" not in chunk.columns:
        chunk["ID"] = 0  # Fill dummy values

    # Encode categoricals
    for col, mapping in encoders.items():
        if col in chunk.columns:
            chunk.loc[:, col] = chunk[col].astype(str).map(mapping).fillna(-1).astype(np.int32)

    # Ensure all expected columns exist
    for col in expected_features:
        if col not in chunk.columns:
            chunk[col] = 0

    X = chunk[expected_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    dtest = xgb.DMatrix(X, enable_categorical=True)
    y_pred_probs = booster.predict(dtest)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true_all.extend(y_true.tolist())
    y_pred_all.extend(y_pred.tolist())

# ==============================
# METRICS
# ==============================
print("\n✅ Evaluation complete — computing metrics...")

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

acc = accuracy_score(y_true_all, y_pred_all)
prec = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
rec = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)

print("\n📊 Evaluation Metrics:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true_all, y_pred_all)
labels = [inv_label_map[i] for i in range(len(inv_label_map))]

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix (XGBoost)")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45, ha="right")
plt.yticks(tick_marks, labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(r"E:\genomics_project\models\xgboost_confusion_matrix.png")
plt.close()

print("🖼️ Confusion matrix saved: xgboost_confusion_matrix.png")

# ==============================
# CLASSIFICATION REPORT
# ==============================
report = classification_report(
    y_true_all,
    y_pred_all,
    target_names=labels,
    zero_division=0,
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(r"E:\genomics_project\models\xgboost_classification_report.csv")
print("📄 Detailed classification report saved: xgboost_classification_report.csv")

print("\n🎉 XGBoost evaluation complete!")

