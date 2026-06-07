import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── CONFIG ────────────────────────────────────────────────────────────────
DATA_PATH   = 'RUL_02_04.csv','RUL_01_03.csv', 'RUL_02.csv', 'RUL_04.csv', 'RUL_03.csv', 'RUL_01.csv', 'RUL_01_02_03_04.csv'
ENGINE_COL  = "engine"
CYCLE_COL   = "cycle"
STAGE_COL   = "cluster"                 # 0-4 degradation stage
TEST_SIZE   = 0.20
SEED        = 42

# ── 1) LOAD & SORT ─────────────────────────────────────────────────────────
df = (pd.read_csv(DATA_PATH)
        .sort_values([ENGINE_COL, CYCLE_COL], ignore_index=True))

# ── 2) FAST RUL CALC (linear time) ─────────────────────────────────────────
def fast_rul(group: pd.DataFrame) -> pd.Series:
    cycles = group[CYCLE_COL].to_numpy()
    stages = group[STAGE_COL].to_numpy()
    n      = len(group)
    rul    = np.zeros(n, dtype=int)

    next_idx_for_stage = {}
    for i in range(n - 1, -1, -1):
        cur = stages[i]
        higher = [next_idx_for_stage[s] for s in next_idx_for_stage if s > cur]
        rul[i] = (cycles[min(higher)] - cycles[i]) if higher else 0
        next_idx_for_stage[cur] = i
    return pd.Series(rul, index=group.index)

df["RUL"] = (
    df[[ENGINE_COL, CYCLE_COL, STAGE_COL]]
      .groupby(ENGINE_COL, group_keys=False, sort=False)
      .apply(fast_rul)
)

# ── 3) BUILD FEATURES (numeric only) ───────────────────────────────────────
drop_cols = {ENGINE_COL, CYCLE_COL, STAGE_COL, "RUL", "source_file"}
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include="number")
y = df["RUL"]

# ── 4) TRAIN / TEST SPLIT ──────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

# ── 5a) FAST HISTOGRAM GBDT REGRESSOR ──────────────────────────────────────
hgb = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.1,
                                    max_iter=300, random_state=SEED).fit(X_tr, y_tr)
y_pred_hgb = hgb.predict(X_te)

# ── 5b) COMPACT RANDOM FOREST REGRESSOR ────────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                           random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)

# ── 6) METRICS (works on any sklearn version) ──────────────────────────────
def report(name, y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:<28} RMSE: {rmse:8.3f} | MAE: {mae:8.3f} | R²: {r2:6.3f}")

print("\n── Regression performance on test set ──")
report("HistGradientBoostingRegressor", y_te, y_pred_hgb)
report("RandomForestRegressor",         y_te, y_pred_rf)
from joblib import dump  # joblib is in the scikit-learn dependency stack

# ── 7) Persist models ────────────────────────────────────────────────────────
dump(hgb, "hist_gradient_boosting_rul.joblib")
dump(rf,  "random_forest_rul.joblib")

print("✔️  Saved:")
print(" • hist_gradient_boosting_rul.joblib")
print(" • random_forest_rul.joblib")
