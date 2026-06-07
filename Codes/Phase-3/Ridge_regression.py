#!/usr/bin/env python
# train_ridge_rul.py
"""
Train a Ridge regression RUL predictor for CMAPSS data (raw sensors only).
• Computes true RUL
• Uses raw sensor readings as features
• Standardizes features and fits Ridge
• Evaluates on hold-out set
• Saves trained pipeline to disk
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_PATH  = Path("RUL\\RUL_01_03.csv")
ENGINE_COL = "engine"
CYCLE_COL  = "cycle"
STAGE_COL  = "cluster"
SEED       = 42
TEST_SIZE  = 0.2

# ── 1) LOAD & SORT ────────────────────────────────────────────────────────
df = (
    pd.read_csv(DATA_PATH)
      .sort_values([ENGINE_COL, CYCLE_COL], ignore_index=True)
)

# ── 2) COMPUTE TRUE RUL ───────────────────────────────────────────────────
def fast_rul(g: pd.DataFrame) -> pd.Series:
    c, s = g[CYCLE_COL].to_numpy(), g[STAGE_COL].to_numpy()
    nxt, rul = {}, np.zeros(len(g), dtype=int)
    for i in range(len(g)-1, -1, -1):
        hi = [nxt[k] for k in nxt if k > s[i]]
        rul[i] = (c[min(hi)] - c[i]) if hi else 0
        nxt[s[i]] = i
    return pd.Series(rul, index=g.index)

# apply fast_rul grouped by engine
grouped = df[[ENGINE_COL, CYCLE_COL, STAGE_COL]].groupby(
    ENGINE_COL, group_keys=False, sort=False
)
if "include_groups" in grouped.apply.__code__.co_varnames:
    df["RUL"] = grouped.apply(fast_rul, include_groups=False)
else:
    df["RUL"] = grouped.apply(fast_rul)

# ── 3) SELECT RAW SENSOR FEATURES ─────────────────────────────────────────
sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
X = df[sensor_cols]
y = df["RUL"]

# ── 4) SPLIT TRAIN/VALID ──────────────────────────────────────────────────
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# ── 5) BUILD & TRAIN RIDGE PIPELINE ─────────────────────────────────────
pipeline = make_pipeline(
    StandardScaler(),
    Ridge(random_state=SEED)
)
pipeline.fit(X_tr, y_tr)

# ── 6) EVALUATE ──────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_va)
rmse = np.sqrt(mean_squared_error(y_va, y_pred))
mae  = mean_absolute_error(y_va, y_pred)
r2   = r2_score(y_va, y_pred)
print(f"Validation → RMSE {rmse:.3f} | MAE {mae:.3f} | R² {r2:.3f}")

# ── 7) SAVE PIPELINE ─────────────────────────────────────────────────────
output_path = "ridge_raw_pipeline.joblib"
dump(pipeline, output_path)
print(f"✓ Saved Ridge pipeline → {output_path}")
