# Contributions — Team T033, Project 26

## Member → Files → Roles

| Member | Enrollment | Files Owned | Phase 1 Role | Phase 2 Role |
|---|---|---|---|---|
| **Kushal Sarkar** | 2401020033 | `README.md`, `requirements.txt`, `.gitignore`, `CONTRIBUTING.md`, `CONTRIBUTIONS.md`, `run_all.py`, `configs/config.yaml`, `src/__init__.py`, `src/config.py`, `src/data.py`, `src/utils.py`, `src/metrics.py`, `tests/test_no_leakage.py`, `tests/test_metrics.py`, `docs/data_card.md`, `docs/viva_notes/kushal_start.md` | Scaffold, shared pipeline (data loading, splits, features, metrics, utils), integration, report assembly | Integration lead, hybrid ablation review, final Phase 2 report |
| **Prachee Dhar** | 2401010330 | `notebooks/01_eda.ipynb`, `reports/ieee_report/sections/p1_literature.tex`, `reports/ieee_report/sections/p1_eda.tex`, `reports/refs_prachee.bib`, `docs/viva_notes/prachee.md` | Literature review (≥12 refs), EDA (duplicates, imbalance, KS drift, Gaussianity, Amount/Time), dataset quality | DL literature review, DL dataset & regularization |
| **Apoorva Choudhary** | 2401010092 | `src/features.py`, `src/gmm_model.py`, `notebooks/02_feature_engineering.ipynb`, `notebooks/03_gmm_baseline.ipynb`, `results/scores/gmm_val.npy`, `results/scores/gmm_test.npy`, `results/tables/fe_*.csv`, `results/tables/gmm_*.csv`, `results/figures/fe_*.png`, `results/figures/gmm_*.png`, `reports/ieee_report/sections/p1_features.tex`, `reports/ieee_report/sections/p1_gmm_theory.tex`, `reports/ieee_report/sections/p1_gmm_results.tex`, `reports/refs_apoorva.bib`, `docs/viva_notes/apoorva.md` | Feature engineering, GMM grid (cov type × K), AIC/BIC, EM theory, instability study | Latent GMM theory (Phase 2) |
| **Vani Rudra** | 2401010490 | `src/baselines.py`, `notebooks/04_classical_baselines.ipynb`, `results/scores/isoforest_*.npy`, `results/scores/ocsvm_*.npy`, `results/scores/lof_*.npy`, `results/scores/pca_recon_*.npy`, `results/tables/baselines_*.csv`, `results/figures/baselines_*.png`, `reports/ieee_report/sections/p1_baselines_theory.tex`, `reports/ieee_report/sections/p1_baselines_results.tex`, `reports/refs_vani.bib`, `docs/viva_notes/vani.md` | Classical anomaly-detection baselines (Isolation Forest, One-Class SVM, LOF, PCA reconstruction) and their theory | Autoencoder design, implementation, reconstruction-score evaluation |
| **Krishna Pramod Dubale** | 2401010239 | `src/evaluation.py`, `notebooks/05_evaluation.ipynb`, `results/tables/eval_*.csv`, `results/figures/eval_*.png`, `reports/ieee_report/sections/p1_evaluation_protocol.tex`, `reports/ieee_report/sections/p1_results_comparison.tex`, `reports/refs_krishna.bib`, `docs/viva_notes/krishna.md` | Evaluation toolkit (bootstrap CI, paired bootstrap diff), threshold policies (FPR budget, review capacity, max-F1), cost curves, master comparison table | Hybrid ablation study, model comparisons, technical validation |

## Notes

- All members read the full project guide and each other's prompt files for context.
- Each member's viva notes in `docs/viva_notes/` serve as individual understanding evidence.
- Commit history on each member's branch (`feat/<name>`) is additional evidence.
- Score files (`.npy`) are tracked because they are small and needed for reproducibility; raw data is NOT committed.
