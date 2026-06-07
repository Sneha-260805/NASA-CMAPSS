# Cluster Classify Predict

## Contributors
Each team member contributed equally (20%) across the following tasks:

| Name     | Contribution                                           |
|----------|--------------------------------------------------------|
| Member 1 | Developed individual models for datasets FD001 & FD002 |
| Member 2 | Developed individual models for datasets FD003 & FD004 |
| Member 3 | Built combined model for datasets FD001 + FD003        |
| Member 4 | Built combined model for datasets FD002 + FD004        |
| Member 5 | Developed the unified model combining all datasets     |

## File Structure
project-root/
├── data/ # Directory for raw FD00x datasets
│ ├── FD001.csv
│ ├── FD002.csv
│ ├── FD003.csv
│ └── FD004.csv
├── models/ # Trained model outputs
│ ├── individual/ # Individual dataset models
│ ├── combined/ # Paired dataset models
│ └── combined_all/ # Unified model for all datasets
├── results/ # Evaluation results and metrics
├── train_individual_models.py # Script to train on single datasets
├── train_combined_model.py # Script to train on combined datasets
├── evaluate_models.py # Script to evaluate trained models
├── requirements.txt # Python dependencies
└── README.md # Project documentation