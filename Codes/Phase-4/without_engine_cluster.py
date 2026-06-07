import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
FILE_PATH = 'clustered_cmapss_data_01_03_with_stage4_flag.csv','clustered_cmapss_data_02_04_with_stage4_flag.csv','clustered_cmapss_data_01_02_03_04_with_stage4_flag.csv','04_04_1.csv','03_04_1.csv','02_04_1.csv','01_04_1.csv'

OUT_PATH  = FILE_PATH.with_name(FILE_PATH.stem + "_no_engine_cluster.csv")

# ── LOAD ──────────────────────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH)

# ── DROP the columns ──────────────────────────────────────────────────────
cols_to_drop = ["engine", "cluster"]           # rename here if needed
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# ── SAVE ──────────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"✔︎ Cleaned file written → {OUT_PATH}")
