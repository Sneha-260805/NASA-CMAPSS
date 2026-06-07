# 

import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
INPUT_FILE = Path(r"Phase-4\\clustered_cmapss_data_01_03_with_RUL_with_risk.csv")
OUTPUT_FILE = INPUT_FILE.with_name(INPUT_FILE.stem + "_engine.csv")

# ── LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

# ── PER-ENGINE MIN–MAX NORMALISE of 'risk' ─────────────────────────────────
# For each engine group, scale risk so its min→0 and max→1.
def normalize_group(x):
    min_val = x.min()
    max_val = x.max()
    if max_val == min_val:
        return 0.0  # or pd.NA if you prefer
    return (x - min_val) / (max_val - min_val)

df["risk_norm"] = df.groupby("engine")["risk"].transform(normalize_group)

# ── SAVE the augmented CSV ───────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f"✔︎ Saved with per-engine normalized risk → {OUTPUT_FILE}")
