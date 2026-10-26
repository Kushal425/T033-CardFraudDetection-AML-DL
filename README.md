# Card Fraud Triage — GMM Density Scoring vs Autoencoder Anomaly Detector

> **Team T033 | Section A | Project 26 | Rishihood University**
> Phase 1 (Machine Learning only) in progress. Phase 2 (Deep Learning) is future work.

## Problem Statement

Credit card fraud is a severe, highly imbalanced classification problem: genuine fraud represents only ~0.172 % of all transactions in the ULB dataset. Rather than treating this as standard binary classification, this project frames it as **anomaly detection**: models are trained exclusively on legitimate transactions and score each new transaction by how unusual it is relative to the learned normal distribution. Phase 1 compares a **Gaussian Mixture Model (GMM)** density scorer against classical baselines (Isolation Forest, One-Class SVM, LOF, PCA reconstruction). Phase 2 (future) will additionally compare an autoencoder reconstruction-error scorer. Fair comparison is enforced by using the same legitimate-only training set and an untouched mixed test set, with thresholds selected on a held-out validation set.

## Dataset

- **Source:** [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group)
- **Scale:** 284,807 transactions across 48 continuous hours (September 2013).
- **Class Imbalance:** 492 fraud cases (base rate: **0.1727%** or 1 fraud per 578 legitimate transactions).
- **Features:** 30 numerical input features (V1–V28 PCA projections for cardholder confidentiality, plus raw `Time` and `Amount`). Target is binary `Class` ($1 = \text{fraud}, 0 = \text{legitimate}$).
- **Data Hygiene:** Strictly non-overlapping chronological partitioning (60% train, 20% validation, 20% test) with scalers fit only on legitimate training records to eliminate data leakage.
- **Local Location:** Expected at `data/raw/creditcard.csv` (excluded from git via `.gitignore`).

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
│   ├── features.py              # feature engineering pipelines
│   ├── gmm_model.py             # GMM training, AIC/BIC, instability study
│   ├── baselines.py             # PCA, Isolation Forest, OC-SVM, LOF baselines
│   ├── evaluation.py            # bootstrap CI, threshold policies, cost curves
│   ├── final.py                 # final model comparisons and paper numbers
│   ├── metrics.py               # all evaluation metrics (single source of truth)
│   └── utils.py                 # seed, logger, score/table/figure I/O
├── tests/
│   ├── test_no_leakage.py       # time ordering, scaler mean, no fraud in legit
│   ├── test_metrics.py          # unit tests for src/metrics.py
│   ├── test_evaluation.py       # evaluation module verification
│   └── test_score_contract.py   # contract tests for all 14 score arrays
├── notebooks/
│   ├── 01_literature_and_eda.ipynb      # EDA and literature
│   ├── 02_feature_engineering.ipynb     # Feature transformations
│   ├── 03_gmm_baseline.ipynb            # GMM density modeling
│   ├── 03b_classical_baselines.ipynb    # Classical anomaly detectors
│   └── 05_evaluation.ipynb              # Evaluation toolkit and curves
├── results/
│   ├── scores/                  # {model}_{split}.npy  (higher = more anomalous)
│   ├── figures/                 # PNG plots (PR curves, ROC, cost, feature importance)
│   ├── tables/                  # CSV metric tables and paper_numbers.tex
│   └── models/                  # serialised models (gmm.joblib, gmm.pkl)
├── reports/
│   ├── T033_Phase1_IEEE_Report.pdf      # Compiled IEEE conference report (8 pages)
│   ├── ieee_report/                     # LaTeX IEEE report source and canonical refs.bib
│   └── literature_review.md             # Standalone literature review deliverable
├── docs/
│   ├── T033_Phase1_Presentation_Slides.pdf # Phase 1 presentation deck (10 slides)
│   ├── data_card.md                     # dataset description and compliance audit
│   ├── FINAL_AUDIT.md                   # integration and verification audit log
│   └── slides_phase1_outline.md         # presentation slide outline
├── guides/
│   ├── PHASE1_ONLY.md                   # Phase 1 project scope and deliverables
│   └── VIVA_GUIDE.md                    # comprehensive master viva preparation guide
├── run_all.py                           # orchestrates Phase 1 stages
├── requirements.txt
└── CONTRIBUTIONS.md                     # verified roles, enrollments, and owned files
```

## Deliverables & Submission Artifacts

- 📄 **IEEE Conference Paper (PDF):** [`reports/T033_Phase1_IEEE_Report.pdf`](reports/T033_Phase1_IEEE_Report.pdf) (8-page double-column IEEEtran format)
- 📊 **Presentation Slides Deck (PDF):** [`docs/T033_Phase1_Presentation_Slides.pdf`](docs/T033_Phase1_Presentation_Slides.pdf) (10-slide 16:9 presentation deck)
- 📝 **LaTeX Source:** [`reports/ieee_report/main_phase1.tex`](reports/ieee_report/main_phase1.tex) with canonical bibliography [`reports/ieee_report/refs.bib`](reports/ieee_report/refs.bib)
- 📦 **Overleaf Bundle Archive:** `overleaf_T033_phase1.zip` (self-contained package for 1-click Overleaf compilation)
- 🎓 **Master Viva Defense Guide:** [`guides/VIVA_GUIDE.md`](guides/VIVA_GUIDE.md) (individual member Q&A, theoretical deep dives, code walkthrough)

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
