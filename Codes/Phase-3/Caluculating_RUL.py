import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
FILE_PATH   = Path("clustered_cmapss_data_01.csv")
OUTPUT_PATH = Path("RUL_01.csv")
ENGINE_COL  = "engine"
CYCLE_COL   = "cycle"
STAGE_COL   = "cluster"   # 5‐stage label from your clustering step

# ── 1) Load & sort ─────────────────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH)
df = df.sort_values([ENGINE_COL, CYCLE_COL]).reset_index(drop=True)

# ── 2) Compute RUL per engine ─────────────────────────────────────────────────
def compute_rul(group: pd.DataFrame) -> pd.DataFrame:
    cycles  = group[CYCLE_COL].values
    stages  = group[STAGE_COL].values
    rul     = np.zeros(len(group), dtype=int)

    # for each point, find the next cycle where stage increases
    for i in range(len(group)):
        # indices after i where stage > current stage
        future = np.where((stages > stages[i]) & (np.arange(len(group)) > i))[0]
        if future.size > 0:
            j = future[0]
            rul[i] = cycles[j] - cycles[i]
        else:
            # no further stage increase → set RUL to 0 (or np.nan if you prefer)
            rul[i] = 0

    group = group.copy()
    group["RUL"] = rul
    return group

df_with_rul = df.groupby(ENGINE_COL, group_keys=False).apply(compute_rul)

# ── 3) Save new dataset ───────────────────────────────────────────────────────
df_with_rul.to_csv(OUTPUT_PATH, index=False)
print(f"Saved dataset with RUL → {OUTPUT_PATH}")
