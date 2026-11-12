# Phase 2 Viva & Oral Defense Notes — Apoorva Choudhary (`P2_02`)

**Role:** Phase 2 Latent-Space Gaussian Mixture Model (Hybrid H1 Theory, Implementation, Diagnostics & Multi-Dim Ablation)
**Primary Code:** `src/hybrid_latent_gmm.py` | **Tests:** `tests/test_hybrid_latent_gmm.py` | **Notebook:** `notebooks/08_latent_gmm.ipynb`
**Official Scores:** `results/scores/hybrid_latentgmm_val.npy`, `results/scores/hybrid_latentgmm_test.npy`

---

## 1. Core Viva Questions & Model Answers

### Q1. Why fit a GMM in the Autoencoder's latent space instead of only in the $d=31$ ambient feature space?
1. **Dimensionality & Parameter Economy:** A `full`-covariance $K$-component GMM in dimension $d$ requires $O(K d^2)$ covariance parameters. At $K=12$, ambient $d=30$ requires $5{,}951$ parameters ($d=31$ requires $6{,}335$; `full` $K=16$ at $d_z=12$ uses $1{,}455$), whereas latent $d_z=8$ `full` $K=12$ drops to $539$ parameters ($91.49\%$ reduction vs $d=31$, $11.0\times$ fewer than $d=30$; at $K=8$, $3{,}967$ at $d=30$ drops to $359$ at $d_z=8$). Our selected `diag` in $d_z=12$ with $K=12$ requires only $299$ free parameters ($21.2\times$ fewer than ambient $d=31$ `full`).
2. **Numerical Conditioning & Convergence Speed:** Across 5 EM seeds ($42\text{--}46$, $n_{\text{init}}=1$, $K=12, \varepsilon=0.01$), latent `diag` EM runs $4.9\times$--$5.9\times$ faster (`1.48--5.72` s vs `7.48--28.25` s for ambient `full`) and maintains covariance condition numbers $\kappa \in [7.62, 8.95]$ instead of $[460.0, 1558.7]$ (ambient `full`) or $8.67 \times 10^8$ (unregularized latent `full` at $\varepsilon=10^{-7}$).
3. **Orthogonal Failure Signal to Reconstruction Error:** Autoencoder reconstruction error detects points *off* the learned manifold, whereas latent GMM detects frauds that map *onto* the manifold cleanly (low reconstruction error) yet land in low-density or underrepresented latent pockets. Empirically, at `FPR = 0.5%`, H1 detects `8` test frauds missed by `ae_best` (raising combined union recall from `0.6800` to `0.7867`) and improves Recall@0.5% FPR by $+8.00$ points over Phase 1 GMM (`0.7467` vs `0.6667`).

### Q2. Write down the exact mathematical pipeline from $x$ to the anomaly score $s(x)$.
1. **Standardized Input & Frozen Encoder:** Given standardized `base` input $\mathbf{x} \in \mathbb{R}^{31}$, Vani's trained encoder produces $\mathbf{z} = f_\phi(\mathbf{x}) \in \mathbb{R}^{d_z}$ ($d_z = 12$ in `ae_best.pt`, `LeakyReLU(0.01)` output activation).
2. **Mixture Density:** Fitted strictly on legitimate training latent vectors $\mathcal{Z}_{\text{train}}^0$ ($N=170{,}524$):
   $$p(\mathbf{z} \mid \Theta) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \qquad \pi_k \ge 0,\; \sum_{k=1}^{K}\pi_k = 1.$$
3. **EM Updates & Covariance Regularization:** E-step computes posterior responsibilities $\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{z}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{z}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$ using log-sum-exp stabilization; M-step updates $\pi_k$, $\boldsymbol{\mu}_k$, and $\boldsymbol{\Sigma}_k^{\text{reg}} = \boldsymbol{\Sigma}_k + \varepsilon \mathbf{I}$ (`reg_covar` $= \varepsilon > 0$), guaranteeing $\lambda_{\min}(\boldsymbol{\Sigma}_k^{\text{reg}}) \ge \varepsilon$ and finite positive log-determinant.
4. **Anomaly Score:** $s(\mathbf{z}) = -\ln p(\mathbf{z} \mid \Theta)$, computed via `score_samples(Z)` negated (`float32`), so higher score strictly means lower likelihood / more anomalous. Because $-\ln p(\mathbf{z})$ is strictly monotonically decreasing in density, any monotone calibration leaves PR-AUC, ROC-AUC, and Recall@FPR invariant.

### Q3. Why did BIC select `full, K=16`, while Validation PR-AUC selected `diag, K=12, reg=0.01`?
- **Density–Anomaly Objective Mismatch:** $\text{BIC} = -2\ell(\hat{\Theta}) + M\ln N$ measures how tightly the mixture fits *legitimate* bulk density on training/val-legit data. A flexible `full` $K=16$ mixture ($1{,}455$ parameters) achieves lower val-legit BIC (`1,217,896` vs `1,716,652`), but tight anisotropic covariances overfit benign sub-clusters and create blind spots or poorly scaled Mahalanobis directions for held-out fraud (`0.05745` val PR-AUC, `0.50877` Rec@0.5%).
- **Validation PR-AUC Selection:** Using validation labels strictly for hyperparameter selection favors `diag, K=12, reg=0.01` ($299$ parameters), boosting selection-split val PR-AUC to `0.15099` and Recall@0.5% FPR to `0.71930`, and keeping condition numbers below `8.31`.

### Q4. Why is Hybrid H1's PR-AUC lower than AE and Phase 1 GMM (`0.1724` vs `0.3574` / `0.2679`), while Recall@0.5% FPR (`0.7467`) beats both?
- **LeakyReLU Bottleneck Non-Gaussianity:** Auditing the 12 latent coordinates of `ae_best.pt` on legitimate training rows shows **0.0%** satisfy mild Gaussianity (`|skew| < 1` and `|kurtosis| < 3`), vs **38.71%** in ambient $d=31$. Every latent axis is heavily right-skewed (`mean |skew| = 5.053`, `max = 11.388`, `mean Shapiro-Wilk W = 0.7135`).
- **Top-Rank False Positive Sensitivity:** Because Gaussian components assume symmetric elliptical tails, legitimate transactions sitting far in the LeakyReLU positive half-line incur large quadratic Mahalanobis penalties (max test NLL `246.35`), flooding the top-50 alert queue (`Precision@50 = 0.02` test, `0.00` val) and depressing global PR-AUC (which heavily weights early precision).
- **Bulk Separation at Operational Budget:** Once past those handful of extreme positive-tail legitimate outliers, fraud scores separate cleanly from normal transactions (`ROC-AUC = 0.9398`, `Recall@0.5% FPR = 0.7467`, catching `56/75` test frauds vs AE's `51/75` and Phase 1 GMM's `50/75`).

---

## 2. Five-Part Diagnostic Evidence Summary

1. **Evidence (i) — Selection Rule (`latentgmm_selection_comparison.csv`):**
   - Train-legit BIC (`full, K=16, reg=1e-6`): val PR-AUC `0.05204`, Rec@0.5% `0.40351`.
   - Val-legit BIC (`full, K=16, reg=1e-2`): val PR-AUC `0.05745`, Rec@0.5% `0.50877`.
   - Val PR-AUC (`diag, K=12, reg=1e-2`): val PR-AUC `0.15099`, Rec@0.5% `0.71930`.
2. **Evidence (ii) — Parent Comparison & Complementarity (`latentgmm_parent_comparison.csv`, `latentgmm_paired_bootstrap.csv`, `latentgmm_complementarity.csv`):**
   - Refit on all `170,524` train-legit rows (`hybrid_latentgmm`): Val PR-AUC `0.1159`, Test PR-AUC `0.1724` `[0.1329, 0.2310]`, Test ROC-AUC `0.9398`, Test Recall@0.5% `0.7467` `[0.6400, 0.8533]`, Test Recall@1.0% `0.7867`, Test Precision@100 `0.3000`.
   - Paired bootstrap differences (`hybrid_latentgmm - parent`) confirm significantly lower PR-AUC (`-0.0955` vs Phase 1 GMM on disk, `-0.1849` vs AE, `-0.3094` vs DAE, `-0.5701` vs PCA) but positive Recall@0.5% deltas over Phase 1 GMM (`+0.0800`) and AE (`+0.0667`), with `9` H1-only test frauds vs GMM and `8` H1-only test frauds vs AE (`78.67%` union recall for both pairs).
3. **Evidence (iii) — Conditioning & Seed Stability (`latentgmm_reg_covar_sensitivity.csv`, `latentgmm_seed_stability.csv`):**
   - At `reg=1e-7`, `full` covariance explodes to $\kappa = 5.27 \times 10^7$ (ambient $d=31$) and $8.67 \times 10^8$ (latent $d=12$); `diag` stays $\kappa \in [6.43, 8.31]$ across all $\varepsilon \in [10^{-7}, 0.1]$.
   - Five seeds (`42..46`): latent `diag` PR-AUC spans `0.1020..0.1695` (`cond 7.62..8.95`, `1.48..5.72 s`), vs ambient `full` `0.2504..0.6429` (`cond 460.0..1558.7`, `7.48..28.25 s`).
4. **Evidence (iv) — Gaussianity & Failure Analysis (`latentgmm_gaussianity_summary.csv`, `latentgmm_failure_analysis.csv`):**
   - Ambient `d=31`: `38.71%` mildly Gaussian, mean Shapiro-Wilk `W=0.8358`.
   - Latent `d=12` (`ae_best`): `0.0%` mildly Gaussian, mean Shapiro-Wilk `W=0.7135`, mean `|skew|=5.053`, mean `|kurtosis|=84.639`.
5. **Evidence (v) — Task C Multi-Dimension Ablation (`ablation_gmm_summary.csv`, 360 runs):**
   - `k=2` (`full, K=16`): test PR-AUC `0.00598 ± 0.00038`, Rec@0.5% `0.0133` (over-compressed).
   - `k=4` (`tied, K=16`): test PR-AUC `0.29198 ± 0.02250`, Rec@0.5% `0.7822` (`full, K=16` `0.27630 ± 0.06268`).
   - `k=8` (`full, K=16`): test PR-AUC `0.01126 ± 0.00156`, Rec@0.5% `0.0222` (fraud signal sits in reconstruction residual rather than latent density, even though AE recon at `k=8` performs well).
   - `k=12` (`ae_best`, `tied, K=16` at `reg=1e-4`): test PR-AUC `0.24425 ± 0.02597`, Rec@0.5% `0.7422`.
   - `k=16` (`tied, K=16` at `reg=1e-4`, `407` params): **val PR-AUC `0.26366 ± 0.03905`, test PR-AUC `0.41490 ± 0.03555`, Rec@0.5% `0.7733`** (`diag, K=16` gets `0.32119` test PR-AUC, `0.7822` Rec@0.5%).

---

## 3. Codebase Walkthrough (`src/hybrid_latent_gmm.py`)

- `WaitingForInputsError` & `check_ae_dependencies()`: Fail-fast dependency validator raising clear missing-input status if Vani's encoder/checkpoints are absent.
- `discover_ae_checkpoints()` & `discover_ablation_checkpoints()`: Locate `ae_best.pt`/`ae_config.json` and `ae_latent_k{2,4,8,16}.pt` dynamically without mutating Vani's files.
- `LatentSplits`, `load_vani_encoder()`, `encode_matrix()`, `encode_splits()`: Load frozen PyTorch AE via Vani's `load_autoencoder()` and extract deterministic latent embeddings for `train_sub_legit`, `train_es_legit`, `train_legit`, `val`, and `test`.
- `fit_latent_gmm()` & `compute_latent_nll()`: Fit `sklearn.mixture.GaussianMixture` strictly on legitimate latent rows and score $-\ln p(\mathbf{z})$ as finite 1-D `float32` vectors.
- `run_latent_gmm_grid_search()` & `select_latent_gmm_config()`: Sweep `covariance_type` × `K` × `reg_covar`, saving AIC/BIC vs validation PR-AUC tables and contrasting BIC vs supervised validation selection.
- `compare_with_parents()`, `compute_parameter_count_comparison()`, `compute_complementarity_summary()`: Evaluate official H1 scores vs Phase 1 GMM, PCA, AE, and DAE with paired bootstrap confidence intervals and `FPR=0.5%` overlap counts.
- `run_reg_covar_sensitivity()`, `run_seed_stability_comparison()`, `analyze_latent_gaussianity()`, `summarize_gaussianity()`, `analyze_latentgmm_failures()`: Execute numerical conditioning, multi-seed EM stability, Shapiro--Wilk/skew/kurtosis Gaussianity, and top-FP/FN audits.
- `run_gmm_ablation()`: Execute Task C's 360-run grid across all 5 bottleneck widths (`k=2,4,8,12,16`), generating `ablation_gmm_runs.csv` and `ablation_gmm_summary.csv`.
- `plot_latent_aic_bic()`, `plot_latent_nll_distribution()`, `plot_latent_stability_evidence()`, `plot_latent_qq()`, `plot_latent_2d_projection()`: Generate all publication figures in `results/figures/latentgmm_*.png` (plus `latentgmm_ablation_dim.png` in `notebooks/08_latent_gmm.ipynb`).
- `run_synthetic_stub_self_test()`, `run_day2_stage()`, `run_day3_stage()`, `run_latent_gmm_stage()`: CLI entry points supporting `--stage {all,day2,day3}` and `--synthetic-stub`.

