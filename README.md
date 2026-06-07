# CCP: Cluster, Classify, Predict
### Hybrid Predictive Maintenance using the NASA CMAPSS Dataset

> **Course:** Machine Learning (CS3102) — Mahindra University, École Centrale School of Engineering  
> **Mentor:** Dr. Neeraj Choudhary  
> **Submitted:** May 8, 2025

---

## Overview

Traditional predictive maintenance models reduce complex mechanical degradation to a binary "working / about to fail" decision — missing the gradual progression that real-world systems undergo. This project proposes a **four-phase hybrid pipeline** that moves beyond binary prediction to model multi-stage engine degradation and produce a real-time, actionable **Risk Score**.

The pipeline is applied to the **NASA CMAPSS turbofan engine dataset** across all four subsets (FD001–FD004) and their combinations.

---

## The CCP Pipeline

```
Raw Sensor Data
      │
      ▼
┌─────────────────────┐
│  Phase 1: CLUSTER   │  Weighted K-Means → 5 degradation stage labels
└─────────────────────┘
      │  (pseudo-labels)
      ▼
┌─────────────────────┐
│  Phase 2: CLASSIFY  │  Random Forest / XGBoost / Logistic Regression
└─────────────────────┘
      │  (stage prediction + failure probability)
      ▼
┌─────────────────────┐
│  Phase 3: PREDICT   │  Random Forest / Ridge / HGB Regressor → stage-wise RUL
└─────────────────────┘
      │  (cycles until next stage)
      ▼
┌─────────────────────┐
│  Phase 4: RISK SCORE│  Risk = Failure Probability × Time-to-Failure → alert if > 0.6
└─────────────────────┘
```

---

## Dataset

The [NASA CMAPSS](https://www.nasa.gov/intelligent-systems-division/) dataset simulates turbofan engine sensor readings (temperatures, pressures, vibration) from run-to-failure. Each engine cycle records **3 operational settings** and **21 sensor measurements**.

| Subset | Operating Conditions | Fault Modes | Training Samples |
|--------|---------------------|-------------|-----------------|
| FD001  | 1 (Sea Level)       | HPC Degradation | 20,631 |
| FD002  | 6                   | HPC Degradation | 53,070 |
| FD003  | 1 (Sea Level)       | HPC + Fan Degradation | 24,710 |
| FD004  | 6                   | HPC + Fan Degradation | 61,984 |

Models were trained and evaluated on individual subsets, merged subsets (FD001∪FD003, FD002∪FD004), and the fully combined dataset (FD001∪FD002∪FD003∪FD004).

---

## Phase 1 — Clustering & Label Generation

Since CMAPSS has no explicit health-stage labels, we used **unsupervised clustering** to generate them.

**Why weighted K-Means?** Standard K-Means and GMM failed to align clusters with the true degradation trajectory because they treat all features equally. We boosted the weight of the **cycle feature** (engine age), making the algorithm sensitive to lifecycle progression. All other features were normalized to [−1, 1].

This produced **5 well-separated degradation stages**:

| Cluster | Stage |
|---------|-------|
| 0 | Normal |
| 1 | Slightly Degraded |
| 2 | Moderately Degraded |
| 3 | Critical |
| 4 | Failure |

Cluster quality was confirmed visually using **PCA** and **UMAP** projections — each cluster mapped to a distinct region of the manifold, with degradation progressing from center (healthy) to periphery (failure).

---

## Phase 2 — Classification

The cluster labels became ground-truth classes for a **supervised multi-class classification** problem.

**Models trained:** Random Forest · XGBoost · Logistic Regression

### Results (Accuracy / F1 / Precision / Recall)

| Dataset | Random Forest | XGBoost | Logistic Regression |
|---------|:---:|:---:|:---:|
| FD001 | 0.986 | 0.987 | 0.984 |
| FD002 | 0.999 | 0.996 | 0.993 |
| FD003 | 0.932 | 0.933 | 0.933 |
| FD004 | 0.999 | 0.996 | 0.996 |
| FD001 ∪ FD003 | 0.996 | 0.995 | 0.987 |
| FD002 ∪ FD004 | 0.999 | 0.996 | 0.995 |
| **Combined All** | **1.000** | **0.996** | **0.997** |

**Random Forest** was the strongest classifier overall. **Logistic Regression**, while slightly lower in accuracy, was essential for producing calibrated failure probabilities used in Phase 4.

---

## Phase 3 — Regression (Stage-wise RUL)

Rather than predicting RUL until absolute failure, we predicted **cycles remaining until the next degradation stage transition** — a more interpretable and actionable target for real-time planning.

**Models trained:** Random Forest Regressor · Ridge Regression · Histogram Gradient Boosting (HGB)

### Results

| Dataset | Ridge R² | RF R² | HGB R² |
|---------|:---:|:---:|:---:|
| FD001 | 0.303 | **0.904** | 0.847 |
| FD002 | 0.291 | **0.923** | 0.879 |
| FD003 | 0.323 | **0.810** | 0.665 |
| FD004 | 0.282 | **0.917** | 0.876 |
| FD001 ∪ FD003 | 0.340 | **0.930** | 0.874 |
| FD002 ∪ FD004 | 0.290 | **0.904** | 0.871 |
| **Combined All** | 0.295 | **0.905** | 0.870 |

**Random Forest Regressor** consistently outperformed Ridge Regression by a large margin (R² ~0.90 vs ~0.30), demonstrating the importance of modeling nonlinear degradation behavior. On the fully combined dataset: MAE = 3.36 cycles, RMSE = 5.55 cycles.

---

## Phase 4 — Composite Risk Score & Decision Logic

The outputs of Phases 2 and 3 are fused into a single, interpretable **Risk Score** per engine cycle:

```
Risk Score = Failure Probability × Time to Stage 4
```

- **Failure Probability** — from Logistic Regression (probability of being in Cluster 4)
- **Time to Stage 4** — stage-wise RUL from the regression model

The raw score is normalized to [0, 1] using Min-Max scaling. By plotting normalized scores against engine cycle for all FD subsets, a consistent pattern emerged: scores crossed **0.6** exactly when engines were entering or approaching the failure stage.

> **Decision Rule:** Any engine with a normalized Risk Score > **0.6** is flagged for proactive maintenance.

This threshold was empirically validated across all dataset configurations and remained robust on merged and fully combined datasets.

---

## Repository Structure

```
├── Codes/
│   ├── Phase-1/                    # Weighted K-Means clustering, UMAP
│   ├── Phase-2/                    # Random Forest, XGBoost, Logistic Regression
│   ├── Phase-3/                    # RFR, Ridge, HGB regressors + RUL prediction
│   └── Phase-4/                    # Risk score, threshold logic, plots
├── Plots/
│   ├── Phase-1/                    # PCA cluster plots, UMAP projections
│   ├── Phase-2/                    # Confusion matrices (all models × all datasets)
│   ├── Phase-3/                    # True vs. Predicted RUL plots
│   └── Phase-4/                    # Failure probability vs. cycle, risk score curves
├── Report/
│   ├── G_AI121_CSE051_AI087_AI097_AI094_Report.pdf
│   └── G_AI121_CSE051_AI087_AI097_AI094_report_source_docx_latex.tex
└── README.md
```

> **Note:** Raw data files (`.txt`, `.csv`) and saved model binaries (`.joblib`, `.pkl`) are excluded via `.gitignore`. Download the CMAPSS dataset from [NASA's Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/).

---

## Dependencies

```
scikit-learn
xgboost
numpy
pandas
matplotlib
umap-learn
joblib
```

Install all at once:
```bash
pip install scikit-learn xgboost numpy pandas matplotlib umap-learn joblib
```

---

## Team

| Name | Roll No. |
|------|----------|
| Sneha Suravajjula | SE23UARI121 |
| Puchalapalli Harika | SE23UARI097 |
| Ananya Pachwa | SE23UARI087 |
| Srija Polisetty | SE23UARI094 |
| Damodaram Lalitha Manasvini | SE23UCSE051 |

---

## Citation

If you use this work, please cite the original dataset:

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," in *2008 International Conference on Prognostics and Health Management*, IEEE, 2008.
