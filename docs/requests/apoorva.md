# Shared Interface Change Requests — Apoorva Choudhary (Task 02)
**Team T033 — Card Fraud Triage — Phase 1**

### Status: No Shared File Changes Required
All Task 02 implementations (Feature Engineering, GMM Baseline, Anomaly Scoring, AIC/BIC Analysis, Instability Studies, IEEE report sections, and viva notes) were completed strictly within owned files without requiring modifications to shared codebase files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, or teammates' files).

### Fulfilled Contracts & Registered Artifacts
1. **Feature Set Registration (`src/features.py`):**
   - Registered `"engineered"` and candidate sets (`"raw"`, `"fe_norm_outliers"`, `"fe_pca_whitened"`, `"fe_robust_amount"`, `"fe_yeojohnson_amount"`, `"fe_time_risk"`, `"fe_val_selected"`) directly into `FEATURE_SETS`.
   - `get_features("engineered")` returns aligned `Splits` dataclass with $d=31$ features, with `StandardScaler` fit exclusively on training legitimate rows (Class == 0).
   - Provides `run_feature_stage()` callable for `run_all.py`.

2. **Anomaly Score Contracts (`src/gmm_model.py`):**
   - Output files:
     - `results/scores/gmm_val.npy` (shape `(56961,)`, dtype `float32`)
     - `results/scores/gmm_test.npy` (shape `(56962,)`, dtype `float32`)
   - Anomaly convention: $s(\mathbf{x}) = -\ln p(\mathbf{x})$ (Negative Log-Likelihood, NLL). Higher score indicates higher suspicion / more anomalous.
   - Serialized model: `results/models/gmm.joblib`.
   - Provides `run_gmm_stage()` callable for `run_all.py`.

3. **Evaluation Protocol:**
   - Evaluated via `src.metrics.evaluate_scores`, producing `results/tables/gmm_metrics.csv`.
   - Thresholds determined exclusively on the Validation split and evaluated forward on the Test split without test snooping.
