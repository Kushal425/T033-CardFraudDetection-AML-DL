> **PHASE 1 = MACHINE LEARNING ONLY.** Do not implement or mention neural networks / autoencoders / deep learning in code, report or notes (a one-sentence "future work" mention is allowed).

# PHASE 1 TASK 03 — Vani Rudra — Classical ML Baselines (Model Application)
Project 26: Card Fraud Triage — Team T033
**Rubric items you cover:** *Model Application* (classical anomaly detectors for fair comparison with GMM), *Theoretical Rigor* (baseline theory), *GitHub & Code Quality*.

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase1/P1_03_Vani_Classical_Baselines.md` completely. Implement ONLY the tasks in it. Only create/edit files under "Files you own". Never edit `src/data.py`, `src/metrics.py` or others' files; put change requests in `docs/requests/vani.md`. Never fabricate numbers. Every model is fit on **training legitimate rows only**; fraud labels are used only for hyper-parameter/threshold selection on VAL and evaluation via `src/metrics.py`; test is touched once. Seed everything. Branch `feat/vani`, commits prefix `[vani]`. No deep learning. At the end write `docs/viva_notes/vani.md` explaining each algorithm in plain English.

## Prerequisite
`git pull origin main` (Kushal's scaffold). Use `from src.data import get_features`, `from src.metrics import evaluate_scores`, `from src.utils import save_scores, save_table, save_fig`.

## Files you own
```
src/baselines.py
notebooks/03b_classical_baselines.ipynb
results/scores/{pca,iforest,ocsvm,lof,random,amount}_{val,test}.npy      # higher = more anomalous
results/models/{pca,iforest,ocsvm,lof}.joblib   results/tables/baseline_*.csv   results/figures/baseline_*.png
reports/ieee_report/sections/p1_baselines.tex
reports/refs_vani.bib
docs/viva_notes/vani.md
```

## Task A — Models (`src/baselines.py`), all on `get_features("base")`
1. **PCA reconstruction error**: fit PCA on train-legit; score = ‖x − x̂‖²; k from explained-variance targets 90/95/99% and a k-sweep; choose k on val PR-AUC. Plot explained variance & PR-AUC vs k.
2. **Isolation Forest**: n_estimators ∈ {100,200,500}, max_samples ∈ {256,1024,'auto'}, max_features ∈ {0.5,1.0}; score = −`score_samples`. Do NOT use contamination for fitting.
3. **One-Class SVM** (RBF): ν ∈ {0.001,0.01,0.05}, γ ∈ {'scale',0.01,0.1}; fit on a seeded 30–50k subsample of train-legit (state the size and why — O(n²)); score = −`decision_function`.
4. **LOF** (`novelty=True`): n_neighbors ∈ {10,20,50}; seeded subsample for fitting; score = −`score_samples`.
5. **References**: random scores (PR-AUC must ≈ base rate — sanity check) and `|Amount|`-only score.
6. Hyper-parameter selection = best **val** PR-AUC (state the rule before running; never test). Report grid table, runtime and memory per model.
7. Evaluate every model via `evaluate_scores` → `results/tables/baseline_metrics.csv` (PR-AUC, ROC-AUC, recall@FPR {0.1,0.5,1}%, precision@review-capacity, top-K).
8. Save scores to the contract paths; hand them to Krishna (evaluation toolkit) & Kushal (final comparison).
9. Diagnostics: score histograms legit vs fraud; top-20 false positives vs caught frauds; which models agree (rank correlation matrix between GMM, PCA, IF, OCSVM, LOF scores); one paragraph of failure analysis.

## Task B — Theory (`p1_baselines.tex`, ~1 page)
- Isolation Forest: random axis-parallel splits, anomalies isolated in fewer splits; score s(x,n)=2^(−E[h(x)]/c(n)).
- PCA reconstruction: projection onto top-k eigenvectors; error = energy in residual subspace; link to Mahalanobis in the minor subspace.
- OC-SVM: separating data from origin in kernel space; ν bounds fraction of outliers/support vectors.
- LOF: local reachability density ratio.
- Compare to GMM: explicit parametric density (GMM) vs geometric/isolation/kernel methods — assumptions and when each fails (high-d, heavy tails, scaling).

## Definition of Done
- [ ] 6 score pairs saved, lengths = split sizes, direction verified. [ ] Grid tables. [ ] Rank-correlation matrix. [ ] LaTeX. [ ] Viva notes. [ ] PR to `main`.

## 🎓 Human: what YOU must be able to say in your viva
Isolation Forest path-length intuition; why OC-SVM needs a subsample; how PCA recon error becomes an anomaly score; why random baseline ≈ base rate; how you chose each hyper-parameter (val only); which baseline was strongest and why.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: make meaningful incremental commits as work is completed (Phase 1).** Do not squash useful history or manipulate dates. Plan the work in the suggested sittings, adapting to the real deadline:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | IsolationForest + PCA baselines + scores |
| 2 | Day 2 | OC-SVM + LOF + references + metrics table |
| 3 | Day 3 | diagnostics, LaTeX, viva notes |
Each commit should be a real, runnable increment, message format `[vani] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/vani`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/vani`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/vani.md` and tell Kushal.
