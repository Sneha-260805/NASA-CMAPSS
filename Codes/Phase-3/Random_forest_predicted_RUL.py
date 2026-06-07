"""
Predict RUL for a **new CMAPSS-style CSV** with the models you saved earlier.
The script
1. loads the saved HistGradientBoostingRegressor (or RF),
2. preprocesses the new data exactly like the training pipeline,
3. optionally computes the *true* RUL (so you can see metrics),
4. adds a column **predicted_RUL**,
5. writes the augmented CSV to disk.
"""

import warnings, sys
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── CONFIG ─────────────────────────────────────────────────────────────────
NEW_DATA_IN   = Path("RUL_ALL\\RUL_01_02_03_04.csv")              # <-- change to your incoming file
NEW_DATA_OUT  = Path("rfr_01_02_03_04_predicted_RUL.csv")
MODEL_PATH    = Path("random_forest_rul.joblib")  # or "random_forest_rul.joblib"

ENGINE_COL  = "engine"
CYCLE_COL   = "cycle"
STAGE_COL   = "cluster"          # 0-4 degradation stage
ID_DROP     = {"source_file"}    # add extra non-numeric ids here if present

# ── 1) LOAD NEW DATA ────────────────────────────────────────────────────────
df = (pd.read_csv(NEW_DATA_IN)
        .sort_values([ENGINE_COL, CYCLE_COL], ignore_index=True))

df = df.drop(['RUL'],axis=1)

# ── 2) OPTIONAL: compute TRUE RUL for evaluation (same O(N) algorithm) ─────
def fast_rul(g: pd.DataFrame) -> pd.Series:
    c, s = g[CYCLE_COL].to_numpy(), g[STAGE_COL].to_numpy()
    nxt, rul = {}, np.zeros(len(g), dtype=int)
    for i in range(len(g) - 1, -1, -1):
        hi = [nxt[k] for k in nxt if k > s[i]]
        rul[i] = c[min(hi)] - c[i] if hi else 0
        nxt[s[i]] = i
    return pd.Series(rul, index=g.index)

gb = df[[ENGINE_COL, CYCLE_COL, STAGE_COL]].groupby(ENGINE_COL, group_keys=False, sort=False)
if "include_groups" in gb.apply.__code__.co_varnames:          # pandas >= 2.2
    df["RUL_true"] = gb.apply(fast_rul, include_groups=False)
else:                                                          # older pandas
    df["RUL_true"] = gb.apply(fast_rul)

# ── 3) BUILD FEATURE MATRIX  (same rules as training) ───────────────────────
drop_cols = {ENGINE_COL, CYCLE_COL, STAGE_COL, "RUL_true"} | ID_DROP
X = (df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        .select_dtypes(include="number"))

# ── 4) LOAD MODEL & PREDICT ─────────────────────────────────────────────────
model = load(MODEL_PATH)
df["predicted_RUL"] = model.predict(X)

# ── 5) OPTIONAL METRICS (only if you know the true RUL) ─────────────────────
if "RUL_true" in df.columns:
    mse  = mean_squared_error(df["RUL_true"], df["predicted_RUL"])
    print("RMSE:", sqrt(mse))
    print("MAE :", mean_absolute_error(df["RUL_true"], df["predicted_RUL"]))
    print("R²  :", r2_score(df["RUL_true"], df["predicted_RUL"]))

# ── 6) SAVE AUGMENTED CSV ───────────────────────────────────────────────────
df.to_csv(NEW_DATA_OUT, index=False)
print(f"✅  Predictions written → {NEW_DATA_OUT}")
