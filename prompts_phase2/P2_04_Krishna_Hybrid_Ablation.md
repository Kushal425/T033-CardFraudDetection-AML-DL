> **PHASE 2 = DEEP LEARNING + HYBRID.** Prerequisite: Phase 1 is merged and tagged `phase1-final`. Do not modify any Phase 1 file or result; build on top of them. Branch name: `feat/<name>-p2`.

# PHASE 2 TASK 04 — Krishna Pramod Dubale — Hybrids H2/H3, Ablation Orchestration, Phase 2 Master Evaluation
**Rubric items you cover:** *Hybrid Innovation* (H2, H3), *Ablation Studies*, *Technical Validation*.

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase2/P2_04_Krishna_Hybrid_Ablation.md` completely. Implement ONLY the tasks in it. Only create/edit files under "Files you own". Never edit shared files or others' files; use `docs/requests/krishna.md`. Never fabricate numbers. Fusion weights and all hyper-parameters are chosen on VAL only; test once. Needs `results/scores/{gmm,ae,ae_dae,hybrid_latentgmm}_*.npy`, `src/autoencoder.py` — fail with a clear message if absent; test with labelled synthetic stubs never saved to `results/`. Reuse your Phase 1 `src/evaluation.py`. Branch `feat/krishna-p2`, prefix `[krishna]`. Never alter commit dates. At the end write `docs/viva_notes/krishna_p2.md`.

## Files you own
```
src/hybrid_fusion.py  src/hybrid_dagmm_lite.py  src/ablation.py
notebooks/09_hybrids_ablation.ipynb
results/scores/hybrid_fusion_{val,test}.npy  hybrid_dagmmlite_{val,test}.npy
results/tables/hybrid_*.csv  ablation_*.csv  eval_master_phase2.csv   results/figures/hybrid_*.png ablation_*.png
reports/ieee_report/sections/p2_hybrid.tex  p2_ablation.tex  p2_validation.tex
reports/refs_krishna_p2.bib   docs/viva_notes/krishna_p2.md
```
## Task A — Hybrids (rationale first: GMM = explicit density but fragile in high-d; AE = flexible compression but no density and can over-generalise — a hybrid must have a *reason*)
- **H2 Score fusion**: normalise GMM-NLL and AE-MSE with train-legit statistics (z-score on log-scores / rank), `s = w·s_gmm + (1−w)·s_ae`, w∈[0,1] step 0.1 on val; also max-fusion and rank-average. Plot w-sensitivity.
- **H3 DAGMM-lite**: concatenate latent z with reconstruction features (MSE, cosine distance) → GMM on [z, err]; score = −log p. (Inspired by Zong et al. 2018 `[VERIFY]`; not end-to-end DAGMM.)
- Compare H1 (Apoorva), H2, H3 vs parents: metrics + bootstrap CI + paired difference; top-K alert **overlap** (Venn), frauds caught by hybrid but missed by both parents and vice-versa; rank correlation GMM vs AE.
## Task B — Ablation (`src/ablation.py`): one factor changed at a time from the final config, 3 seeds where stochastic, val AND test PR-AUC (select on val only)
Feature set raw/base/engineered · GMM cov-type & K (from Apoorva's CSV) · AE latent 2/4/8/16 · AE type undercomplete/denoising σ · AE regularization (from Vani's CSV) · hybrid none/H1/H2/H3 · fusion w · training-data size 10/25/50/100% of train-legit · with/without cyclic time · Amount transform raw/log1p/robust. One paragraph per factor: what changed, what we learned, significant (CI)?
## Task C — Technical validation (`p2_validation.tex`)
5-seed mean±std (from Vani), paired bootstrap results, sanity checks (random baseline ≈ base rate; shuffled-label check; train/val/test drift), runtime/parameter table, threshold-policy table for all final models (val-frozen thresholds), cost curve for the best two.
## Task D — Master table `eval_master_phase2.csv`: GMM, PCA, IsolationForest, OC-SVM, LOF, AE, DAE, H1, H2, H3 with CIs.
## Definition of Done
- [ ] H2 & H3 scores. [ ] Overlap analysis. [ ] Ablation table + plots. [ ] Master table. [ ] LaTeX. [ ] Viva notes. [ ] PR.
## 🎓 Viva
Why each hybrid should help; how w was chosen; what the ablation showed (incl. negative results); how the paired bootstrap supports/doesn't support a win.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: at least 3 meaningful commits on at least 3 different calendar days (Phase 2).** Do not squash. Plan three sittings:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | H2 fusion + stub tests |
| 2 | Day 2 | H3 DAGMM-lite + overlap analysis |
| 3 | Day 3 | ablation table, master table, LaTeX, viva notes |
Each commit = a real, runnable increment, message format `[krishna] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/krishna-p2`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/krishna-p2`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/krishna.md` and tell Kushal.
