#!/usr/bin/env python
# predict_rul_ridge.py
"""
Predict RUL for a new CMAPSS-style CSV using a saved Ridge pipeline.
• Loads the Ridge pipeline (with scaler + model) from disk
• Computes true RUL for evaluation (optional)
• Uses raw sensor readings as features
• Adds RUL_true and predicted_RUL at the end of the CSV
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── CONFIG ─────────────────────────────────────────────────────────────────
NEW_DATA_IN   = Path("RUL\\RUL_01_03.csv")
NEW_DATA_OUT  = Path("ridge_01_03_predicted_RUL.csv")
PIPELINE_PATH = Path("ridge_raw_pipeline.joblib")  # your saved Ridge pipeline

ENGINE_COL  = "engine"
CYCLE_COL   = "cycle"
STAGE_COL   = "cluster"

# ── 1) LOAD & SORT ────────────────────────────────────────────────────────
df = pd.read_csv(NEW_DATA_IN).sort_values([ENGINE_COL, CYCLE_COL], ignore_index=True)

# Drop any existing RUL column
df = df.drop(columns=[c for c in ["RUL"] if c in df.columns])

# ── 2) COMPUTE TRUE RUL ───────────────────────────────────────────────────
def fast_rul(g: pd.DataFrame) -> pd.Series:
    c, s = g[CYCLE_COL].to_numpy(), g[STAGE_COL].to_numpy()
    nxt, rul = {}, np.zeros(len(g), dtype=int)
    for i in range(len(g)-1, -1, -1):
        hi = [nxt[k] for k in nxt if k > s[i]]
        rul[i] = (c[min(hi)] - c[i]) if hi else 0
        nxt[s[i]] = i
    return pd.Series(rul, index=g.index)

gb = df[[ENGINE_COL, CYCLE_COL, STAGE_COL]].groupby(ENGINE_COL, group_keys=False, sort=False)
if "include_groups" in gb.apply.__code__.co_varnames:
    df["RUL_true"] = gb.apply(fast_rul, include_groups=False)
else:
    df["RUL_true"] = gb.apply(fast_rul)

# ── 3) SELECT RAW SENSOR FEATURES ─────────────────────────────────────────
sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
X = df[sensor_cols]

# ── 4) LOAD RIDGE PIPELINE & PREDICT ───────────────────────────────────────
pipeline = load(PIPELINE_PATH)
df["predicted_RUL"] = pipeline.predict(X)

# ── 5) EVALUATE (optional) ─────────────────────────────────────────────────
if "RUL_true" in df.columns:
    mse = mean_squared_error(df["RUL_true"], df["predicted_RUL"])
    print(f"RMSE: {sqrt(mse):.3f}")
    print(f"MAE : {mean_absolute_error(df['RUL_true'], df['predicted_RUL']):.3f}")
    print(f"R²  : {r2_score(df['RUL_true'], df['predicted_RUL']):.3f}")

# ── 6) REORDER COLUMNS & SAVE ─────────────────────────────────────────────
base = [c for c in df.columns if c not in ["RUL_true", "predicted_RUL"]]
ordered = base + ["RUL_true", "predicted_RUL"]
df = df[ordered]

df.to_csv(NEW_DATA_OUT, index=False)
print(f"✅ Predictions written → {NEW_DATA_OUT}")
