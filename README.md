# Card Fraud Triage — GMM Density Scoring vs Autoencoder Anomaly Detector

> **Team T033 | Section A | Project 26 | Rishihood University**  
> Complete Phase 1 (Classical ML) and Phase 2 (Deep Learning & Hybrid Models) Implementation & Evaluation.

## Problem Statement

Credit card fraud is a severe, highly imbalanced classification problem: genuine fraud represents only ~0.172 % of all transactions in the ULB benchmark. Rather than treating this as standard binary classification, this project frames it as **anomaly detection**: models are trained exclusively on legitimate transactions and score each new transaction by how unusual it is relative to the learned normal distribution.
- **Phase 1 (Classical ML):** Evaluated Gaussian Mixture Model (GMM) density scoring against classical anomaly baselines (Isolation Forest, One-Class SVM, LOF, PCA reconstruction).
- **Phase 2 (Deep Learning & Hybrids):** Extended evaluation to Deep Undercomplete Autoencoders (AE), Denoising Autoencoders (DAE), Latent-Space Gaussian Mixture Density (H1), Normalized Score Fusion (H2), and Decoupled DAGMM-lite (H3).

Fair comparison is enforced by strictly using the same legitimate-only training set ($N=170,524$), a held-out validation set ($N=56,961$) for tuning and early stopping, and an untouched chronological test split ($N=56,962$, 75 frauds) evaluated once.

## Empirical Results Summary (Held-Out Test Set)

| Model Key | Display Name | Paradigm | Test PR-AUC | 95% Bootstrap CI | Recall @ 0.1% FPR | Recall @ 0.5% FPR | Precision @ K=100 |
|---|---|---|---|---|---|---|---|
| `pca` | **PCA Reconstruction** | Phase 1 Classical | **0.7425** | [0.6415, 0.8368] | **76.00%** | 77.33% | **57.0%** |
| `ae_dae` | **Denoising Autoencoder (DAE)** | Phase 2 Deep Learning | **0.4818** | [0.3707, 0.6166] | 70.67% | 77.33% | 53.0% |
| `hybrid_fusion` | **H2: Score Fusion** | Phase 2 Hybrid | **0.4818** | [0.3707, 0.6166] | 70.67% | 77.33% | 53.0% |
| `hybrid_dagmmlite`| **H3: DAGMM-lite** | Phase 2 Hybrid | **0.4541** | [0.3524, 0.5884] | 68.00% | **80.00%** | 48.0% |
| `lof` | **Local Outlier Factor** | Phase 1 Classical | 0.4478 | [0.3447, 0.5822] | 65.33% | 72.00% | 47.0% |
| `ae` | **Standard Deep AE** | Phase 2 Deep Learning | 0.3574 | [0.2618, 0.4848] | 53.33% | 68.00% | 40.0% |
| `gmm` | **Full-Space GMM** | Phase 1 Classical | 0.2679 | [0.1843, 0.3800] | 33.33% | 66.67% | 31.0% |
| `hybrid_latentgmm`| **H1: Latent GMM** | Phase 2 Hybrid | 0.1724 | [0.1329, 0.2310] | 24.00% | 74.67% | 30.0% |
| `ocsvm` | **One-Class SVM** | Phase 1 Classical | 0.1322 | [0.1030, 0.1784] | 0.00% | 69.33% | 16.0% |
| `iforest` | **Isolation Forest** | Phase 1 Classical | 0.0577 | [0.0444, 0.0787] | 1.33% | 34.67% | 7.0% |

**Key Research Takeaways:**
1. **DAE significantly outperforms full-space GMM** ($\Delta = +0.2139$, $p < 0.05$), doubling fraud recall at tight operational budgets ($70.67\%$ vs $33.33\%$).
2. **Linear PCA remains the strongest standalone detector** ($0.7425$ PR-AUC) due to the PCA-derived coordinate system of the benchmark and non-linear over-generalization in deep networks.
3. **DAGMM-lite (H3) achieves the highest operational catch rate** at FPR $\le 0.5\%$ across all models (**80.00% recall**, catching 60 of 75 test frauds).

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
| Prachee Dhar | 2401010330 | Literature review, EDA, dataset quality | Hybrid model (H2/H3), ablation, master evaluation |
| Apoorva Choudhary | 2401010092 | Feature engineering, GMM grid & theory | Latent GMM theory |
| Vani Rudra | 2401010490 | Classical anomaly-detection baselines | Autoencoder implementation |
| Krishna Pramod Dubale | 2401010239 | Evaluation toolkit, bootstrap CIs, threshold analysis | DL literature review, DL data pipeline |

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

### Phase 2 Deliverables (Deep Learning & Hybrids)
- 📝 **Phase 2 IEEE Report Source:** [`reports/ieee_report/main_phase2.tex`](reports/ieee_report/main_phase2.tex) (with canonical bibliography [`reports/ieee_report/refs_phase2.bib`](reports/ieee_report/refs_phase2.bib))
- 📄 **Phase 2 Inlined Standalone LaTeX Source:** [`reports/ieee_report/main_phase2_standalone.tex`](reports/ieee_report/main_phase2_standalone.tex)
- 📦 **Phase 2 Overleaf Bundle Archive:** `overleaf_T033_phase2.zip` (self-contained package with all figures and sections for 1-click Overleaf compilation)
- 📓 **Phase 2 Final Comparison Notebook:** [`notebooks/06b_final_comparison_phase2.ipynb`](notebooks/06b_final_comparison_phase2.ipynb) (executed top-to-bottom with 10-model comparisons, bootstrap CIs, and paired differences)
- 📊 **Phase 2 Presentation Slides Outline:** [`docs/slides_phase2_outline.md`](docs/slides_phase2_outline.md) (15-slide high-density conference outline with speaker allocations)
- 🔍 **Phase 2 Final Integration Audit Log:** [`docs/FINAL_AUDIT_P2.md`](docs/FINAL_AUDIT_P2.md) (comprehensive PASS/FAIL checklist, commit cadence disclosure, and contract verification)
- 🎓 **Phase 2 Lead Viva Defense Notes:** [`docs/viva_notes/kushal_final_p2.md`](docs/viva_notes/kushal_final_p2.md) (theoretical deep dives, Q17–Q39 viva preparation, and operational triage recommendations)

### Phase 1 Deliverables (Classical ML Baseline Archive)
- 📄 **Phase 1 IEEE Conference Paper (PDF):** [`reports/T033_Phase1_IEEE_Report.pdf`](reports/T033_Phase1_IEEE_Report.pdf) (8-page double-column IEEEtran format)
- 📊 **Phase 1 Presentation Slides Deck (PDF):** [`docs/T033_Phase1_Presentation_Slides.pdf`](docs/T033_Phase1_Presentation_Slides.pdf) (10-slide 16:9 presentation deck)
- 📝 **Phase 1 LaTeX Source:** [`reports/ieee_report/main_phase1.tex`](reports/ieee_report/main_phase1.tex) with canonical bibliography [`reports/ieee_report/refs.bib`](reports/ieee_report/refs.bib)
- 📦 **Phase 1 Overleaf Bundle Archive:** `overleaf_T033_phase1.zip`
- 🎓 **Phase 1 Master Viva Defense Guide:** [`aml dl prompts/guides/VIVA_GUIDE.md`](aml%20dl%20prompts/guides/VIVA_GUIDE.md)

## How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/Kushal425/T033-CardFraudDetection-AML-DL.git
cd T033-CardFraudDetection-AML-DL

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Place the dataset (excluded from git via .gitignore)
#    Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    Place at: data/raw/creditcard.csv (284,807 rows, 31 cols)

# 4. Run full automated test suite (139 passing tests)
pytest tests/ -v

# 5. Run the Phase 2 master integration and empirical comparison pipeline
python3 src/final_phase2.py

# 6. Run Phase 1 pipeline stages (optional)
python3 run_all.py
```

## Badges

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Phase](https://img.shields.io/badge/phase-1%20ML%20only-blue)

## ⚠️ Disclaimer

> **Not for production fraud decisions.** This project is an academic exercise.
> The models are trained on a publicly available anonymised dataset and have
> not been validated for deployment in any real financial system.
