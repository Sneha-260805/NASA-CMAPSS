#!/usr/bin/env python
# train_logreg_stage4_prob.py
# ------------------------------------------------------------
# Logistic-regression → probability output  +  model persistence
# ------------------------------------------------------------

import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ───────────── CONFIG ─────────────────────────────────────────────────────
DATA_FILE   = 'clustered_cmapss_data_01_03_with_stage4_flag_no_engine_cluster.csv','clustered_cmapss_data_02_04_with_stage4_flag_no_engine_cluster.csv','clustered_cmapss_data_01_02_03_04_with_stage4_flag_no_engine_cluster.csv','01_04_1_no_engine_cluster.csv','02_04_1_no_engine_cluster.csv','03_04_1_no_engine_cluster.csv','04_04_1_no_engine_cluster.csv' #update accordingly
MODEL_FILE  = DATA_FILE.with_name("logreg_stage4_prob.joblib")
OUT_CSV     = DATA_FILE.with_name(DATA_FILE.stem + "_with_lr_prob.csv")

TARGET_COL  = "stage4_flag"        # change if your label has a different name
TEST_SIZE   = 0.20                 # 20 % held-out evaluation split
RANDOM_SEED = 42

# ───────────── LOAD ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)


if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

X = df.drop(columns=[TARGET_COL, "cycle_norm"])
y = df[TARGET_COL].astype(int)

# ───────────── TRAIN / VALIDATE ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

pipeline = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",     # helpful if stage-4 is rare
            solver="lbfgs",
            random_state=RANDOM_SEED
        ))
    ]
)

pipeline.fit(X_train, y_train)

print("\n── Validation metrics on 20 % hold-out ──")
y_prob_val = pipeline.predict_proba(X_test)[:, 1]
y_pred_val = (y_prob_val >= 0.5).astype(int)

print(classification_report(y_test, y_pred_val, digits=3))
print("Confusion-matrix:\n", confusion_matrix(y_test, y_pred_val))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_val):.3f}")

# ───────────── SAVE MODEL ─────────────────────────────────────────────────
dump(pipeline, MODEL_FILE)
print(f"\n✔︎ Model saved → {MODEL_FILE}")

# ───────────── PREDICT FULL DATASET ───────────────────────────────────────
df["lr_prob"] = pipeline.predict_proba(X)[:, 1]

# ───────────── WRITE CSV ──────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"✔︎ Augmented CSV with lr_prob written → {OUT_CSV}")
