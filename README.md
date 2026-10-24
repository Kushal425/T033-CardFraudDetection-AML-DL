# Card Fraud Triage — GMM Density Scoring vs Autoencoder Anomaly Detector

> **Team T033 | Section A | Project 26 | Rishihood University**
> Phase 1 (Machine Learning only) in progress. Phase 2 (Deep Learning) is future work.

## Problem Statement

Credit card fraud is a severe, highly imbalanced classification problem: genuine fraud represents only ~0.172 % of all transactions in the ULB dataset. Rather than treating this as standard binary classification, this project frames it as **anomaly detection**: models are trained exclusively on legitimate transactions and score each new transaction by how unusual it is relative to the learned normal distribution. Phase 1 compares a **Gaussian Mixture Model (GMM)** density scorer against classical baselines (Isolation Forest, One-Class SVM, LOF, PCA reconstruction). Phase 2 (future) will additionally compare an autoencoder reconstruction-error scorer. Fair comparison is enforced by using the same legitimate-only training set and an untouched mixed test set, with thresholds selected on a held-out validation set.

## Team

| Name | Enrollment | Phase 1 Role | Phase 2 Role |
|---|---|---|---|
| Kushal Sarkar | 2401020033 | Scaffold, splits, metrics, integration, report assembly | Integration, hybrid ablation |
| Prachee Dhar | 2401010330 | Literature review, EDA, dataset quality | DL literature, dataset regularization |
| Apoorva Choudhary | 2401010092 | Feature engineering, GMM grid & theory | Latent GMM theory |
| Vani Rudra | 2401010490 | Classical anomaly-detection baselines | Autoencoder implementation |
| Krishna Pramod Dubale | 2401010239 | Evaluation toolkit, bootstrap CIs, threshold analysis | Hybrid ablation, comparisons |

Faculty: Shobhit Johri · Arun Chauhan · Ashwin Tewary (Rishihood University)

## Repository Structure

```
T033-CardFraudDetection-AML-DL/
├── configs/config.yaml          # seed, paths, split fractions, eval budgets
├── src/
│   ├── config.py                # loads config.yaml
│   ├── data.py                  # load_raw, chronological_split, get_features
│   ├── metrics.py               # all evaluation metrics (single source of truth)
│   ├── utils.py                 # seed, logger, score/table/figure I/O
│   ├── features.py              # (Apoorva) engineered feature set
│   ├── gmm_model.py             # (Apoorva) GMM training, AIC/BIC, scores
│   └── evaluation.py            # (Krishna) bootstrap CI, threshold policies
├── tests/
│   ├── test_no_leakage.py       # time ordering, scaler mean, no fraud in legit
│   └── test_metrics.py          # unit tests for src/metrics.py
├── notebooks/
│   ├── 01_eda.ipynb             # (Prachee) EDA and literature
│   ├── 02_feature_engineering.ipynb  # (Apoorva)
│   ├── 03_gmm_baseline.ipynb    # (Apoorva)
│   ├── 04_classical_baselines.ipynb  # (Vani)
│   └── 05_evaluation.ipynb      # (Krishna)
├── results/
│   ├── scores/                  # {model}_{split}.npy  (higher = more anomalous)
│   ├── figures/                 # PNG plots
│   ├── tables/                  # CSV metric tables
│   └── models/                  # serialised models (joblib)
├── reports/ieee_report/         # LaTeX IEEE report sections
├── docs/
│   ├── data_card.md             # dataset description and limitations
│   ├── viva_notes/              # individual viva preparation notes
│   └── requests/                # change requests for shared files
├── run_all.py                   # orchestrates Phase 1 stages
├── requirements.txt
├── CONTRIBUTING.md
└── CONTRIBUTIONS.md
```

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/Kushal425/T033-CardFraudDetection-AML-DL.git
cd T033-CardFraudDetection-AML-DL

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Place the dataset (NOT committed to git)
#    Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    Place at: data/raw/creditcard.csv  (284,807 rows, 31 cols)

# 4. Run the scaffold (split + feature checks)
python -c "from src.data import get_features; s=get_features(); print(s.X_val.shape)"

# 5. Run all Phase 1 stages
python run_all.py

# 6. Run tests
pytest tests/ -v
```

## Badges

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Phase](https://img.shields.io/badge/phase-1%20ML%20only-blue)

## ⚠️ Disclaimer

> **Not for production fraud decisions.** This project is an academic exercise.
> The models are trained on a publicly available anonymised dataset and have
> not been validated for deployment in any real financial system.
