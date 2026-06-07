import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
INPUT_FILE = Path(r"C:\\Users\\sneha\\OneDrive\\Desktop\\AIML_Project\\phase-4\\clustered_04_with_RUL_with_risk_engine.csv")

# ── LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

# ── FILTER: Engine 1 & stage ≤ 3 ─────────────────────────────────────────
df_eng1_stage3 = df[(df["engine"] == 1) & (df["cluster"] <= 4)]

# ── PLOT ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(df_eng1_stage3["cycle"], df_eng1_stage3["risk_norm"], marker="o", linestyle="-")
plt.xlabel("Cycle")
plt.ylabel("Normalized Risk")
plt.title("Engine 1: Normalized Risk vs. Cycle (Up to Stage 4)")
plt.grid(True)
plt.tight_layout()
plt.show()
