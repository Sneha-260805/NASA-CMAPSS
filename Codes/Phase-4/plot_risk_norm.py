import pandas as pd
import matplotlib.pyplot as plt

# List of file paths
file_paths = [
    "Phase-4//clustered_cmapss_data_01_03_with_RUL_with_risk_engine.csv",
    "Phase-4/clustered_cmapss_data_02_04_with_RUL_with_risk_engine.csv"
]

# Corresponding titles for plots
titles = ["Clustered Data FD001+FD003", "Clustered Data FD002+FD004"]

# Loop through files and plot
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    
    # Filter rows where engine == 1
    df_engine_1 = df[df['engine'] == 1]
    
    # Plot cycle vs lr_prob
    plt.figure(figsize=(8, 5))
    plt.plot(df_engine_1['cycle'], df_engine_1['lr_prob'], marker='o', linestyle='-')
    plt.title(f'{titles[i]} - Engine ID 1')
    plt.xlabel('Cycle')
    plt.ylabel('failure probability')
    plt.grid(True)
    
    # Save plot
    plt.savefig(f'plot_cluster_{i+1}.png')
    plt.show()      
    plt.close()