# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from joblib import dump
# from config import OUTPUT_DIR, MODEL_DIR

# import warnings
# warnings.filterwarnings("ignore")

# def train_xgboost():
#     print("📂 Loading combined genomic dataset...")
#     data_path = OUTPUT_DIR / "training_dataset.csv"
#     df = pd.read_csv(data_path)

#     print(f"✅ Dataset loaded with shape: {df.shape}")

#     # Select numeric columns for features
#     X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Response_Type_encoded"], errors="ignore")
#     y = df["Response_Type_encoded"] if "Response_Type_encoded" in df.columns else df["Response_Type"]

#     # Encode text if necessary
#     if y.dtype == "object":
#         y = y.astype("category").cat.codes

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"📊 Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

#     # Define XGBoost model
#     model = xgb.XGBClassifier(
#         n_estimators=300,
#         max_depth=6,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         tree_method="hist"
#     )

#     print("🚀 Training XGBoost model...")
#     model.fit(X_train, y_train)

#     print("✅ Training complete. Evaluating model...")
#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     print(f"🎯 Accuracy: {acc:.4f}")
#     print(classification_report(y_test, y_pred))

#     # Save model
#     model_path = MODEL_DIR / "xgboost_model.joblib"
#     dump(model, model_path)
#     print(f"💾 XGBoost model saved to: {model_path}")

# if __name__ == "__main__":
#     train_xgboost()


# 

# import os
# import pandas as pd
# import xgboost as xgb

# # === CONFIG ===
# DATA_PATH = r"E:\genomics_project\outputs\training_dataset.csv"
# MODEL_OUTPUT = r"E:\genomics_project\models\xgboost_model.json"
# CHUNK_SIZE = 200000  # adjust smaller if you still get memory pressure
# NUM_ROUNDS = 100

# # === TRAIN FUNCTION ===
# def train_xgboost_chunked():
#     print(f"📂 Starting chunked training on {DATA_PATH}")

#     # prepare an empty model (will be updated iteratively)
#     booster = None
#     chunk_num = 0

#     for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False, dtype=str):
#         chunk_num += 1
#         print(f"🧩 Processing chunk {chunk_num} ...")

#         # Drop rows with missing Response_Type
#         chunk = chunk.dropna(subset=["Response_Type"])

#         # Encode target (Response_Type)
#         chunk["Response_Type_encoded"] = chunk["Response_Type"].astype("category").cat.codes
#         y = chunk["Response_Type_encoded"]

#         # Select numeric features automatically
#         X = chunk.select_dtypes(include=["number"]).copy()

#         # Create DMatrix for XGBoost
#         dtrain = xgb.DMatrix(X, label=y)

#         # XGBoost parameters
#         params = {
#             "objective": "multi:softprob",
#             "eval_metric": "mlogloss",
#             "num_class": len(chunk["Response_Type_encoded"].unique()),
#             "tree_method": "hist",  # memory efficient
#             "max_depth": 6,
#             "eta": 0.1
#         }

#         # Train incrementally
#         booster = xgb.train(
#             params=params,
#             dtrain=dtrain,
#             num_boost_round=NUM_ROUNDS,
#             xgb_model=booster  # continue training from previous model
#         )

#         print(f"✅ Finished training on chunk {chunk_num}")

#         # Save checkpoint after each chunk
#         checkpoint_path = MODEL_OUTPUT.replace(".json", f"_chunk{chunk_num}.json")
#         booster.save_model(checkpoint_path)
#         print(f"💾 Checkpoint saved: {checkpoint_path}")

#     # Save final model
#     booster.save_model(MODEL_OUTPUT)
#     print(f"🎯 Final model saved to {MODEL_OUTPUT}")

# if __name__ == "__main__":
#     train_xgboost_chunked()


"""
Chunked XGBoost training script for pharmacogenomic dataset
Author: Nethra + ChatGPT
"""

# import os
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from pathlib import Path
# import joblib

# # ===== PATHS =====
# BASE_DIR = Path("E:/genomics_project")
# DATA_PATH = BASE_DIR / "outputs" / "training_dataset.csv"
# MODEL_PATH = BASE_DIR / "models" / "xgboost_model_chunked.json"
# LABEL_MAP_PATH = BASE_DIR / "models" / "label_map.pkl"

# # ===== SETTINGS =====
# CHUNK_SIZE = 200_000   # number of rows to load per chunk (adjust as per RAM)
# NUM_ROUNDS = 100       # total boosting rounds
# MODEL_DIR = BASE_DIR / "models"
# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# # ===== FUNCTION =====
# def train_xgboost_chunked():
#     print(f"📂 Loading training dataset from {DATA_PATH}")
#     if not DATA_PATH.exists():
#         raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

#     # --- Step 1: Find unique labels across a small preview ---
#     print("🔍 Scanning dataset to detect target classes...")
#     preview = pd.read_csv(DATA_PATH, nrows=50000, low_memory=False)
#     if "Response_Type" not in preview.columns:
#         raise KeyError("❌ 'Response_Type' column not found in dataset!")

#     label_map = {label: idx for idx, label in enumerate(sorted(preview["Response_Type"].dropna().unique()))}
#     num_classes = len(label_map)
#     print(f"✅ Detected {num_classes} classes: {label_map}")

#     # Save label map
#     joblib.dump(label_map, LABEL_MAP_PATH)

#     # --- Step 2: XGBoost parameters ---
#     params = {
#         "objective": "multi:softprob",
#         "num_class": num_classes,
#         "eval_metric": "mlogloss",
#         "learning_rate": 0.1,
#         "max_depth": 6,
#         "subsample": 0.8,
#         "colsample_bytree": 0.8,
#         "base_score": 0.5,    # ✅ Fix for base_score error
#         "verbosity": 1,
#         "tree_method": "hist"  # ✅ memory efficient
#     }

#     booster = None
#     chunk_no = 1

#     # --- Step 3: Train on each chunk ---
#     for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False, dtype=str):
#         print(f"\n🧩 Processing chunk {chunk_no} ... ({len(chunk)} rows)")

#         # Drop rows without Response_Type
#         chunk = chunk.dropna(subset=["Response_Type"])

#         # Encode target
#         y = chunk["Response_Type"].map(label_map)
#         if y.isna().any():
#             print("⚠️ Some unknown Response_Type values found — skipping them.")
#             valid_mask = ~y.isna()
#             chunk = chunk[valid_mask]
#             y = y[valid_mask]

#         # Drop non-numeric / irrelevant columns
#         X = chunk.select_dtypes(include=[np.number]).copy()
#         if X.empty:
#             print("⚠️ No numeric columns found in this chunk, skipping.")
#             continue

#         dtrain = xgb.DMatrix(X, label=y)

#         if booster is None:
#             booster = xgb.train(
#                 params=params,
#                 dtrain=dtrain,
#                 num_boost_round=NUM_ROUNDS
#             )
#         else:
#             booster = xgb.train(
#                 params=params,
#                 dtrain=dtrain,
#                 num_boost_round=NUM_ROUNDS,
#                 xgb_model=booster
#             )

#         print(f"✅ Finished chunk {chunk_no}, continuing training...")
#         chunk_no += 1

#         # Save intermediate progress
#         booster.save_model(MODEL_PATH)
#         print(f"💾 Model checkpoint saved at {MODEL_PATH}")

#     print(f"\n🎉 Training completed successfully on all chunks!")
#     booster.save_model(MODEL_PATH)
#     print(f"✅ Final XGBoost model saved at: {MODEL_PATH}")
#     print(f"📁 Label map saved at: {LABEL_MAP_PATH}")

# # ===== MAIN =====
# if __name__ == "__main__":
#     train_xgboost_chunked()


"""
Chunked XGBoost training with automatic categorical encoding
Author: Nethra + ChatGPT
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib

# ===== PATHS =====
BASE_DIR = Path("E:/genomics_project")
DATA_PATH = BASE_DIR / "outputs" / "training_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model_chunked.json"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map.pkl"
ENCODER_MAP_PATH = BASE_DIR / "models" / "feature_encoders.pkl"

CHUNK_SIZE = 200_000
NUM_ROUNDS = 100

# ===== FUNCTION =====
def train_xgboost_chunked():
    print(f"📂 Loading training dataset from {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

    # Step 1: Identify target labels
    print("🔍 Scanning dataset to detect target classes...")
    preview = pd.read_csv(DATA_PATH, nrows=50000, low_memory=False)
    if "Response_Type" not in preview.columns:
        raise KeyError("❌ 'Response_Type' column not found in dataset!")

    label_map = {label: idx for idx, label in enumerate(sorted(preview["Response_Type"].dropna().unique()))}
    num_classes = len(label_map)
    joblib.dump(label_map, LABEL_MAP_PATH)
    print(f"✅ Detected {num_classes} classes: {label_map}")

    # Step 2: Initialize XGBoost parameters
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "base_score": 0.5,
        "verbosity": 1,
        "tree_method": "hist"
    }

    booster = None
    encoders = {}
    chunk_no = 1

    # Step 3: Train incrementally
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        print(f"\n🧩 Processing chunk {chunk_no} ... ({len(chunk)} rows)")

        chunk = chunk.dropna(subset=["Response_Type"])
        y = chunk["Response_Type"].map(label_map)

        # Select potential feature columns
        features = [c for c in chunk.columns if c not in ["Response_Type"]]

        X = chunk[features].copy()

        # Encode categorical (string) columns to numeric IDs
        for col in X.columns:
            if X[col].dtype == "object":
                if col not in encoders:
                    encoders[col] = {v: i for i, v in enumerate(X[col].dropna().unique())}
                X[col] = X[col].map(encoders[col]).fillna(-1)

        # Convert to DMatrix
        X = X.select_dtypes(include=[np.number])
        if X.empty:
            print("⚠️ No valid numeric features found after encoding, skipping.")
            chunk_no += 1
            continue

        dtrain = xgb.DMatrix(X, label=y)

        # Train model incrementally
        if booster is None:
            booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=NUM_ROUNDS)
        else:
            booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=NUM_ROUNDS, xgb_model=booster)

        booster.save_model(MODEL_PATH)
        print(f"💾 Model checkpoint saved after chunk {chunk_no}")
        chunk_no += 1

    # Save final model and encoders
    if booster:
        booster.save_model(MODEL_PATH)
        joblib.dump(encoders, ENCODER_MAP_PATH)
        print(f"\n🎉 Training complete!")
        print(f"✅ Final model saved: {MODEL_PATH}")
        print(f"📁 Label map saved: {LABEL_MAP_PATH}")
        print(f"📁 Feature encoders saved: {ENCODER_MAP_PATH}")
    else:
        print("⚠️ No data was trained due to missing numeric/categorical encodable features.")

# ===== MAIN =====
if __name__ == "__main__":
    train_xgboost_chunked()
