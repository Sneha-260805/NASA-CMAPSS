import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# ---------- 1. Load ----------
FILE = "train_combined_FD001_FD003.txt"         # update path if needed
df = pd.read_csv(FILE, sep=r"\s+", header=None).dropna(axis=1)

df.columns = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}"     for i in range(1, 22)]
)

# ---------- 2. Prepare ----------
X = df.drop("engine_id", axis=1).copy()
X["cycle"] *= 10.0                               # 10× weight on cycle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 3. Agglomerative Clustering ----------
agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
df["cluster"] = agg.fit_predict(X_scaled)

# ---------- 4. Relabel → stages ----------
cycle_means = df.groupby("cluster")["cycle"].mean().sort_values()
stage_map = {clust_id: stage for stage, clust_id in enumerate(cycle_means.index)}
df["stage"] = df["cluster"].map(stage_map)

# ---------- 5. 2-D projection for plotting ----------
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)
df[["PC1", "PC2"]] = proj

# ---------- 6. Plot ----------
fig, ax = plt.subplots(figsize=(9, 6))

# colour palette (simple greys → hot)
stage_colors = {0: "#4CAF50",   # green  (normal)
                1: "#FFC107",   # amber  (slightly degraded)
                2: "#FF9800",   # orange (moderately degraded)
                3: "#F44336",   # red    (critical)
                4: "#9C27B0"}   # purple (failure)

for stage, grp in df.groupby("stage"):
    ax.scatter(grp["PC1"], grp["PC2"],
               s=12, alpha=0.6,
               label=f"Stage {stage} ({len(grp)})",
               c=stage_colors[stage])

# annotate each cluster’s centroid with mean cycle
for stage in sorted(stage_colors):
    subset = df[df["stage"] == stage]
    x_c, y_c = subset[["PC1", "PC2"]].mean(axis=0)
    mean_cycle = subset["cycle"].mean() / 10      # divide by 10 to undo weighting
    ax.text(x_c, y_c,
            f"Stage {stage}\nμ cycle ≈ {mean_cycle:.0f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.8,
                      edgecolor=stage_colors[stage]))

ax.set_title("Agglomerative Clusters of Engines FD001 + FD003 (PCA view, weighted cycles)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend(title="Clusters / Stages")
ax.grid(lw=0.2, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------- 7. Save enriched data ----------
df.drop(columns=["cluster", "PC1", "PC2"]).to_csv(
    "train_combined_FD001_FD003_with_stages.csv", index=False)
print("✓ Plot displayed and CSV saved → train_combined_FD001_FD003_with_stages.csv")
