# Contributions — Team T033, Project 26

## Member → Files → Roles

| Member | Enrollment | Files Owned | Phase 1 Role | Phase 2 Role |
|---|---|---|---|---|
| **Kushal Sarkar** | 2401020033 | `README.md`, `requirements.txt`, `.gitignore`, `CONTRIBUTIONS.md`, `run_all.py`, `configs/config.yaml`, `src/__init__.py`, `src/config.py`, `src/data.py`, `src/utils.py`, `src/metrics.py`, `src/final.py`, `tests/`, `docs/data_card.md`, `docs/FINAL_AUDIT.md`, `docs/slides_phase1_outline.md`, `guides/VIVA_GUIDE.md` | Scaffold, shared pipeline (data loading, splits, features, metrics, utils), integration, final report assembly | Integration lead, hybrid ablation review, final Phase 2 report |
| **Prachee Dhar** | 2401010330 | `notebooks/01_literature_and_eda.ipynb`, `src/eda.py`, `reports/literature_review.md`, `reports/ieee_report/sections/p1_related_work.tex`, `reports/ieee_report/sections/p1_dataset_eda.tex` | Literature review (≥12 refs), EDA (duplicates, imbalance, KS drift, Gaussianity, Amount/Time), dataset quality audit | DL literature review, DL dataset & regularization |
| **Apoorva Choudhary** | 2401010092 | `src/features.py`, `src/gmm_model.py`, `notebooks/02_feature_engineering.ipynb`, `notebooks/03_gmm_baseline.ipynb`, `results/scores/gmm_*.npy`, `results/tables/fe_*.csv`, `results/tables/gmm_*.csv`, `results/figures/fe_*.png`, `results/figures/gmm_*.png`, `results/models/gmm.*`, `reports/ieee_report/sections/p1_features.tex`, `reports/ieee_report/sections/p1_gmm_theory.tex`, `reports/ieee_report/sections/p1_gmm_results.tex` | Feature engineering, GMM grid (cov type × K), AIC/BIC, EM theory, instability study | Latent GMM theory (Phase 2) |
| **Vani Rudra** | 2401010490 | `src/baselines.py`, `notebooks/03b_classical_baselines.ipynb`, `results/scores/iforest_*.npy`, `results/scores/ocsvm_*.npy`, `results/scores/lof_*.npy`, `results/scores/pca_*.npy`, `results/tables/baseline_*.csv`, `results/figures/baseline_*.png`, `reports/ieee_report/sections/p1_baselines.tex` | Classical anomaly-detection baselines (Isolation Forest, One-Class SVM, LOF, PCA reconstruction) and their theory | Autoencoder design, implementation, reconstruction-score evaluation |
| **Krishna Pramod Dubale** | 2401010239 | `src/evaluation.py`, `notebooks/05_evaluation.ipynb`, `results/tables/eval_*.csv`, `results/figures/eval_*.png`, `reports/ieee_report/sections/p1_evaluation_protocol.tex`, `reports/ieee_report/sections/p1_results_comparison.tex` | Evaluation toolkit (bootstrap CI, paired bootstrap diff), threshold policies (FPR budget, review capacity, max-F1), cost curves, master comparison table | Hybrid ablation study, model comparisons, technical validation |

## Notes

- All members read the project guides and prompt files for context and alignment.
- Individual understanding and member-specific deep dives are consolidated in `guides/VIVA_GUIDE.md`.
- Shared canonical bibliography is maintained in `reports/ieee_report/refs.bib`.
- Git commit logs on each member's feature branch provide verified evidence of contribution.
- Score files (`.npy`) and result tables (`.csv`) are tracked to guarantee reproducibility; raw transaction datasets are strictly ignored.
