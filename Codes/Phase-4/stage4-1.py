import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
FILE_PATH = 'clustered_cmapss_data_01_03.csv','clustered_cmapss_data_02_04.csv','clustered_cmapss_data_01_02_03_04.csv','clustered_01.csv','cclustered_02.csv','clustered_03.csv','clustered_04.csv'

# Output is written next to the input file
OUT_PATH = FILE_PATH.with_name(FILE_PATH.stem + "_with_stage4_flag.csv")

# ── LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH)

# ── 1) Choose the label column name ---------------------------------------
# If you renamed it to "stage", change the next line accordingly.
LABEL_COL = "cluster"          # or "stage"

# ── 2) Create binary flag --------------------------------------------------
df["stage4_flag"] = (df[LABEL_COL] == 4).astype(int)
#  → rows where cluster == 4  → 1
#  → rows where cluster != 4  → 0

# ── 3) Save ----------------------------------------------------------------
df.to_csv(OUT_PATH, index=False)
print(f"✔︎ File with binary flag written → {OUT_PATH}")
