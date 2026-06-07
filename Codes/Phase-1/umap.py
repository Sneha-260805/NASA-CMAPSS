import pandas as pd
import umap
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np # Import numpy for handling potential NaN values

# Define the path to your CSV file
csv_file_path = 'clustered_cmapss_data_01_02_03_04.csv','clustered_cmapss_data_01.csv','clustered_cmapss_data_02.csv','clustered_cmapss_data_03.csv','clustered_cmapss_data_04.csv','clustered_cmapss_data_01_03.csv','clustered_cmapss_data_02_04.csv'

# Define the feature columns to be used for UMAP
feature_columns = [
    'op_setting_1', 'op_setting_2', 'op_setting_3', 'sensor_1', 'sensor_2',
    'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
    'sensor_19', 'sensor_20', 'sensor_21'
]

# Define the target column (clusters)
cluster_column = 'cluster'

# Load the dataset
try:
    print(f"Loading data from {csv_file_path}...")
    data_df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading the CSV file: {e}")
    exit()

# Drop rows with NaN values in feature columns or cluster column, as UMAP cannot handle them
data_df.dropna(subset=feature_columns + [cluster_column], inplace=True)

# Separate features (X) and target (y)
X = data_df[feature_columns]
y = data_df[cluster_column]

print(f"Number of data points: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Unique cluster labels: {np.sort(y.unique())}")

if X.empty:
    print("Error: No data left after removing NaN values. Cannot proceed with UMAP.")
    exit()

# Initialize UMAP.
# You can experiment with n_neighbors, min_dist, and n_components.
# n_neighbors: Controls how UMAP balances local versus global structure in the data.
# min_dist: Controls how tightly UMAP is allowed to pack points together.
# random_state: Ensures reproducibility.
print("Initializing UMAP reducer...")
reducer = umap.UMAP(n_neighbors=30,  # Increased n_neighbors as per typical usage for larger datasets
                   min_dist=0.1,
                   n_components=2,
                   random_state=42,
                   verbose=True)

# Fit and transform the data
print("Fitting UMAP model and transforming data...")
embedding = reducer.fit_transform(X)
print("UMAP transformation complete.")

# Create a scatter plot of the UMAP projection
print("Creating scatter plot...")
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y,
    cmap='Spectral',  # 'Spectral' is a good colormap for categorical data
    s=5  # Smaller marker size for potentially many points
)

# Add a legend
# Since cluster labels might be numerous, create a legend with unique cluster labels
unique_labels = np.sort(y.unique())
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=plt.cm.Spectral(i / (len(unique_labels) -1 if len(unique_labels) > 1 else 1) ), markersize=8)
           for i, label in enumerate(unique_labels)]
plt.legend(handles=handles, title='Cluster')


plt.title('UMAP Projection of Clustered Data', fontsize=16)
plt.xlabel('UMAP Component 1', fontsize=12)
plt.ylabel('UMAP Component 2', fontsize=12)
plt.gca().set_aspect('equal', 'datalim') # Ensure aspect ratio is equal
plt.grid(True)
print("Displaying plot...")
# plt.show() # Comment out or remove plt.show()
plt.savefig('umap_projection.png') # Save the plot to a file
print("Plot saved as umap_projection.png")
print("Script finished.")