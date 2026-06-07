#!/usr/bin/env python
# make_risk_column.py
# -------------------------------------------------------------
# risk  =  P(stage-4 | features)  ×  RUL_to_stage4
# -------------------------------------------------------------

import pandas as pd
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────
RUL_FILE   = 'clustered_cmapss_data_01_03_with_RUL.csv','clustered_cmapss_data_02_04_with_RUL.csv','clustered_cmapss_data_01_02_03_04_with_RUL.csv','clustered_01_with_RUL.csv','clustered_02_with_RUL.csv','clustered_03_with_RUL.csv','clustered_04_with_RUL.csv'
PROB_FILE  = 'clustered_cmapss_data_01_03_with_stage4_flag_no_engine_cluster_with_lr_prob.csv','clustered_cmapss_data_02_04_with_stage4_flag_no_engine_cluster_with_lr_prob.csv','clustered_cmapss_data_01_02_03_04_with_stage4_flag_no_engine_cluster_with_lr_prob.csv','logistic_regression_01.csv','logistic_regression_02.csv','logistic_regression_03.csv','logistic_regression_04.csv'
OUT_FILE   = RUL_FILE.with_name(RUL_FILE.stem + "_with_risk.csv")

RUL_COL    = "RUL_to_stage4"      # column in RUL_FILE
PROB_COL   = "lr_prob"            # column in PROB_FILE  (edit if different)

# ── 1. Load both files ───────────────────────────────────────
df_rul  = pd.read_csv(RUL_FILE)
df_prob = pd.read_csv(PROB_FILE)

# sanity check
if len(df_rul) != len(df_prob):
    raise ValueError(
        f"Row-count mismatch: {len(df_rul)} (RUL file) vs {len(df_prob)} (prob file)"
    )

# ── 2. Add the probability column to the RUL frame ───────────
df_rul[PROB_COL] = df_prob[PROB_COL].values   # assumes identical row order
# If you have common keys (e.g. engine & cycle), merge on those instead:
# df_rul = df_rul.merge(df_prob[[KEYS + [PROB_COL]]], on=KEYS)

# ── 3. Compute risk  =  probability × RUL ─────────────────────
df_rul["risk"] = df_rul[PROB_COL] * df_rul[RUL_COL]

# ── 4. Save ──────────────────────────────────────────────────
df_rul.to_csv(OUT_FILE, index=False)
print(f"✔︎ Risk column added → {OUT_FILE}")
