import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA          # NEW
import matplotlib.pyplot as plt                # NEW
import warnings

# Ignore future warnings from sklearn about n_init
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')

# ── Configuration ────────────────────────────────────────────────────────────
file_path        = 'train_FD001.txt','train_FD002.txt','train_FD003.txt','train_FD004.txt' # update if needed
output_file_path = 'clustered_cmapss_data_01.csv'
plot_file        = 'fd001_clusters_pca.png'     # image will be saved here
n_clusters       = 5
cycle_weight     = 3.3

# ── Load data ────────────────────────────────────────────────────────────────
print(f"Loading data from: {file_path}")
column_names = ['engine', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]
df = pd.read_csv(file_path, sep=r'\s+', header=None, names=column_names)
print(f"Data loaded. Shape: {df.shape}")

# ── 1. Normalise cycle per engine (0-1) ──────────────────────────────────────
df['cycle_norm'] = df.groupby('engine')['cycle'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) else 0
)

# ── 2. Build clustering matrix ──────────────────────────────────────────────
df_cluster = df.drop(columns='engine').copy()

cols_to_norm = [c for c in df_cluster.columns if c not in ['cycle', 'cycle_norm']]
scaler = MinMaxScaler()
df_cluster[cols_to_norm] = scaler.fit_transform(df_cluster[cols_to_norm])

# drop constant cols (except cycle_norm) and raw cycle
const_mask = df_cluster.var() <= 1e-10
const_mask.loc['cycle_norm'] = False
df_cluster = df_cluster.loc[:, ~const_mask]
df_cluster = df_cluster.drop(columns='cycle')

# weight cycle_norm
df_cluster['cycle_norm'] *= cycle_weight

# ── 3. K-means ──────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
raw_labels = kmeans.fit_predict(df_cluster)
print("K-means complete.")

# reorder clusters by average cycle_norm
df['raw_cluster'] = raw_labels
cycle_means = df.groupby('raw_cluster')['cycle_norm'].mean().sort_values()
label_map = {old: new for new, old in enumerate(cycle_means.index)}
df['cluster'] = df['raw_cluster'].map(label_map).astype(int)
df = df.drop(columns='raw_cluster')

# ── 4. Save CSV & counts ────────────────────────────────────────────────────
df.to_csv(output_file_path, index=False)
print(f"Clustered data → {output_file_path}")
print("\nPoints per cluster:\n", df['cluster'].value_counts().sort_index())

# ── 5. 2-D PCA projection & scatter plot ────────────────────────────────────
print("Running PCA (2 components) for visualisation …")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(df_cluster)

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                 c=df['cluster'], cmap='tab10', s=8)
plt.title('FD001 – K-means (k=5) clusters in PCA space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
cbar = plt.colorbar(sc, ticks=range(n_clusters))
cbar.set_label('Cluster')

plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.show()                     # comment out if running headless/remote
print(f"PCA scatter plot saved → {plot_file}")
