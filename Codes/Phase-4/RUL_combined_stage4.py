import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
# Point to the input file ▸ use a *raw string* (r"...") or forward-slashes
FILE_PATH = 'clustered_cmapss_data_01_03.csv','clustered_cmapss_data_02_04.csv','clustered_cmapss_data_01_02_03_04.csv','clustered_01.csv','clustered_02.csv','clustered_03.csv','clustered_04.csv'

# The output file will be called  clustered_01_with_RUL.csv  in the same folder
OUT_PATH = FILE_PATH.with_name(FILE_PATH.stem + "_with_RUL.csv")

# ── LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH)

# Make sure types are as expected
df["engine"]  = df["engine"].astype(int)
df["cycle"]   = df["cycle"].astype(int)
df["cluster"] = df["cluster"].astype(int)

# ── 1) Locate the failure point (cluster==4) for each engine ──────────────
#    This returns a Series indexed by engine:  engine_id → cycle_of_stage4
stage4_cycle = (
    df[df["cluster"] == 4]
    .groupby("engine")["cycle"]
    .max()                 # the first time the engine is in stage-4 is also the max cycle
    .rename("stage4_cycle")
)

# ── 2) Merge that lookup back onto the main frame ────────────────────────
df = df.merge(stage4_cycle, left_on="engine", right_index=True, how="left")

# ── 3) Compute remaining useful life until stage-4 ───────────────────────
df["RUL_to_stage4"] = df["stage4_cycle"] - df["cycle"]

# If an engine has NOT yet reached stage-4 in the dataset, stage4_cycle is NaN
# → leave RUL as <NA> so downstream logic can decide what to do
df.loc[df["stage4_cycle"].isna(), "RUL_to_stage4"] = pd.NA

# ── 4) Clean up helper column & save ─────────────────────────────────────
df = df.drop(columns=["stage4_cycle"])
df.to_csv(OUT_PATH, index=False)

print(f"✔︎ Augmented file written to:  {OUT_PATH}")
