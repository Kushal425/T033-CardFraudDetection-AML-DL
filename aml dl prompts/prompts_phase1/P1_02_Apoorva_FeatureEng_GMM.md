> **PHASE 1 = MACHINE LEARNING ONLY.** Do not implement or mention neural networks / autoencoders / deep learning in code, report or notes (a one-sentence "future work" mention is allowed).

# TASK 02 — Apoorva Choudhary — Feature Engineering, GMM Baseline & Theory
Project 26: Card Fraud Triage (GMM vs Autoencoder) — Team T033
**Rubric items you cover:** Phase 1 → *Feature Engineering*, *Theoretical Rigor*, *Model Application* (GMM).

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase1/P1_02_Apoorva_FeatureEng_GMM.md` completely. Implement ONLY the tasks in it. Only create/edit files under "Files you own". Never edit `src/data.py`, `src/metrics.py` or others' files; put change requests in `docs/requests/apoorva.md`. Never fabricate numbers; every result must come from executed code. Fit scaler/GMM on **training legitimate rows only**; fraud labels are used only for threshold choice/evaluation on val/test through `src/metrics.py`. Branch `feat/apoorva`, small commits with prefix `[apoorva]`. At the end write `docs/viva_notes/apoorva.md` explaining the EM algorithm, AIC/BIC and each function in plain English.

## Prerequisite
`git pull origin main`. Use `from src.data import get_features`, `from src.metrics import evaluate_scores`, `from src.utils import save_scores, save_table, save_fig`.

## Files you own
```
src/features.py            # registers feature set "engineered" into FEATURE_SETS (do not edit "base")
src/gmm_model.py
notebooks/02_feature_engineering.ipynb
notebooks/03_gmm_baseline.ipynb
results/scores/gmm_val.npy  results/scores/gmm_test.npy        # contract: higher = more anomalous (= -log-likelihood)
results/tables/fe_*.csv  results/tables/gmm_*.csv  results/figures/fe_*.png  results/figures/gmm_*.png
reports/ieee_report/sections/p1_features.tex
reports/ieee_report/sections/p1_gmm_theory.tex
reports/ieee_report/sections/p1_gmm_results.tex
reports/refs_apoorva.bib
docs/viva_notes/apoorva.md
```

## Task A — Feature engineering (`src/features.py`, notebook 02)
Be honest in the report: V1–V28 are already PCA outputs, so FE is *limited and principled*, not extensive.
Compare feature sets (each evaluated by the SAME GMM config via val PR-AUC; report table):
1. `raw`: V1–V28 + Amount + Time (scaled).
2. `base` (from Kushal): V1–V28 + log1p(Amount) + sin/cos time.
3. `engineered` candidates (register the best as `engineered`):
   - robust scaling (median/IQR) of Amount vs log1p vs Box-Cox/Yeo-Johnson (fit on train legit).
   - **PCA-whitening** of V's (fit on train legit) — and note V's are already decorrelated.
   - Norm features: ‖V‖₂, number of |V_i|>3 (outlier count), max |V_i|.
   - Hour-bucket one-hot vs cyclic encoding; Amount-bin indicator.
   - Optionally feature selection by dropping V's whose legit-vs-fraud separation is ~0 **using only the VAL set for the decision, documented** (this uses val labels → state it clearly as a model-selection step, never test).
4. Show for each: dimensionality, val PR-AUC, val recall@FPR=0.5%, training time. Pick the winner; document why; **do not use test to choose**.
Add a leakage statement: every transform fit on train-legit only.

## Task B — GMM model (`src/gmm_model.py`, notebook 03)
Use `sklearn.mixture.GaussianMixture`.
1. **Model selection grid**: `covariance_type ∈ {full, tied, diag, spherical}`, `n_components K ∈ {1,2,3,4,5,6,8,10,12,16}`, `reg_covar ∈ {1e-6,1e-4,1e-2}`, `n_init=3`, `init_params='kmeans'`, `max_iter=300`. If slow use a seeded 50k-row subsample of train-legit for the grid, then refit the chosen config on all train-legit.
2. Record **AIC and BIC** (computed on train-legit; also compute on held-out val-legit log-likelihood) + number of free parameters (verify against the formula in Task C). Plot AIC/BIC vs K per covariance type; mark minima; discuss the elbow.
3. Selection protocol — report BOTH: (a) unsupervised choice = argmin BIC; (b) validation choice = best val PR-AUC. If they differ, discuss. Final model = whichever is preselected in `configs/config.yaml` → do not tune on test.
4. Anomaly score = `-gmm.score_samples(X)` (negative log-likelihood). Save `gmm_val.npy`, `gmm_test.npy`, and the fitted model via joblib to `results/models/gmm.joblib`.
5. Evaluate with `src.metrics.evaluate_scores` (PR-AUC, ROC-AUC, recall@FPR {0.1,0.5,1}%, precision@review-capacity, top-K). Save `results/tables/gmm_metrics.csv`.
6. Diagnostics: histogram of NLL for legit vs fraud (val & test, log-y); component weights & means summary; per-component Mahalanobis contribution; which features drive the top-20 highest-scored frauds vs the top-20 false positives; **failure analysis** — 10 missed frauds and why (score near legit mass?).
7. **Instability study (the "Identify the gap" evidence)**: show (i) singular/ill-conditioned covariances → effect of `reg_covar`; (ii) variance of val PR-AUC across 5 seeds; (iii) how performance and BIC behave as dimension grows (fit on first 5, 10, 20, 30 features). This documents the limitations of GMM scoring (frame it as limitations, not as motivation for deep learning).
8. Drift sensitivity: PR-AUC on val vs test (chronological); comment.

## Task C — Theory section (`p1_gmm_theory.tex`, also summarised in notebook markdown)
Write rigorous but readable derivations (LaTeX equations):
- Mixture density p(x)=Σ_k π_k N(x|μ_k,Σ_k); latent variable z; responsibilities γ_{nk}.
- **EM**: E-step γ_{nk}=π_k N(x_n|μ_k,Σ_k)/Σ_j π_j N(x_n|μ_j,Σ_j); M-step N_k=Σγ_{nk}, μ_k, Σ_k, π_k=N_k/N. Explain why log-likelihood is monotonically non-decreasing (Jensen / ELBO) and why it converges only to a **local** optimum (hence `n_init`).
- Singularity problem (component collapsing on one point → |Σ|→0, likelihood→∞) and the `reg_covar` fix.
- Number of free parameters: full = K·d + K·d(d+1)/2 + (K−1); diag = K·d + K·d + (K−1); spherical = K·d + K + (K−1); tied = K·d + d(d+1)/2 + (K−1). Plug d for our feature set.
- AIC = 2p − 2 ln L̂; BIC = p ln n − 2 ln L̂; why BIC penalises more and prefers simpler models; BIC consistency vs AIC predictive.
- Anomaly score = −ln p(x): why a low-density point is "unusual", and the limitation: *density ≠ fraud*; curse of dimensionality; Gaussian assumption (link to Prachee's gaussianity table); heavy-tailed `Amount`.

## Task D — LaTeX
`p1_features.tex`, `p1_gmm_theory.tex`, `p1_gmm_results.tex` (tables from `results/tables`, figures from `results/figures`).

## Definition of Done
- [ ] Scores files exist with correct lengths. [ ] AIC/BIC plot + table. [ ] Feature-set comparison table. [ ] Instability study.
- [ ] `engineered` registered and `get_features("engineered")` works. [ ] Viva notes. [ ] PR to `main`.

## 🎓 Human: what YOU must be able to say in your viva
Derive EM on a whiteboard; explain why log-likelihood never decreases; AIC vs BIC formulas and which K/cov-type won and why; why `reg_covar` is needed; why V1–V28 limit feature engineering; why NLL is a valid anomaly score and when it fails; how many parameters your final GMM has.

---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, rebase date rewriting, or history rewriting). Commits must carry the true time they were made.

**Goal: make multiple meaningful commits as work is completed.** Do not create empty commits or manipulate dates; use the suggested sittings as a planning aid, not a fixed calendar-day requirement:
| Sitting | Day | Commit content (one or more commits) |
|---|---|---|
| 1 | Day 1 | src/features.py + feature-set comparison |
| 2 | Day 2 | GMM grid, AIC/BIC, scores saved |
| 3 | Day 3 | instability study, theory LaTeX, viva notes |
Each commit must be a real, runnable increment with a clear message (`[apoorva] add ...`). Don't make empty or whitespace commits to hit the count.

**No-conflict rules**
1. Edit ONLY the files listed under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on your own branch `feat/apoorva`. Before every push run: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/apoorva`.
3. Open a PR to `main`; Kushal merges. Merge from the PR page (not force-push). Never `git push --force`.
4. Notebooks: only the owner edits their notebook; clear outputs of other people's notebooks never.
5. If you need a change in a shared file, write it to `docs/requests/apoorva.md` and tell Kushal.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: make meaningful incremental commits as work is completed (Phase 1).** Do not squash useful history or manipulate dates. Plan the work in the suggested sittings, adapting to the real deadline:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | src/features.py + feature-set comparison |
| 2 | Day 2 | GMM grid, AIC/BIC, scores saved |
| 3 | Day 3 | instability study, theory LaTeX, viva notes |
Each commit should be a real, runnable increment, message format `[apoorva] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/apoorva`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/apoorva`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/apoorva.md` and tell Kushal.
