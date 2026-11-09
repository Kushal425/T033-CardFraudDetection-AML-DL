> **PHASE 2 = DEEP LEARNING + HYBRID.** Prerequisite: Phase 1 is merged and tagged `phase1-final`. Do not modify any Phase 1 file or result; build on top of them. Branch name: `feat/<name>-p2`.

# PHASE 2 TASK 02 — Apoorva Choudhary — Latent-Space GMM (Hybrid H1), Hybrid Theory & GMM Ablations
**Rubric items you cover:** *Hybrid Innovation* (H1), *Theoretical Rigor*, *Ablation Studies* (GMM factors).

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase2/P2_02_Apoorva_LatentGMM_Theory.md` completely. Implement ONLY the tasks in it. Only create/edit files under "Files you own". Never edit shared files or others' files; use `docs/requests/apoorva.md`. Never fabricate numbers. GMM on latents is fit on **train-legit codes only**; K/covariance chosen by BIC on train-legit and val PR-AUC (val only); test once. Needs Vani's `src/autoencoder.py` (`encode()`) and trained AE weights in `results/models/` — if absent, fail with a clear "waiting for inputs" message; test with labelled synthetic stubs never saved to `results/`. Branch `feat/apoorva-p2`, prefix `[apoorva]`. Never alter commit dates. At the end write `docs/viva_notes/apoorva_p2.md`.

## Files you own
```
src/hybrid_latent_gmm.py   notebooks/08_latent_gmm.ipynb
results/scores/hybrid_latentgmm_{val,test}.npy     # higher = more anomalous (−log p(z))
results/tables/latentgmm_*.csv  results/tables/ablation_gmm_*.csv  results/figures/latentgmm_*.png
reports/ieee_report/sections/p2_hybrid_theory.tex
reports/ieee_report/sections/p2_latent_gmm_results.tex
reports/refs_apoorva_p2.bib   docs/viva_notes/apoorva_p2.md
```
## Task A — H1 Latent-GMM
Encode train-legit/val/test with Vani's chosen AE (and, for the ablation, latent dims 2/4/8/16). Fit `GaussianMixture` on train-legit codes; grid cov-type × K with AIC/BIC (as in Phase 1); score = −log p(z). Report val/test metrics through `evaluate_scores`; compare with Phase 1 GMM (raw features) and the AE alone using paired bootstrap (Krishna's `src/evaluation.py`).
## Task B — Why it should work (evidence, not claims)
(i) Parameter count: full-cov GMM at d=30 vs d=8 — show the numbers; (ii) condition numbers / `reg_covar` sensitivity on raw vs latent; (iii) seed variance of PR-AUC raw-GMM vs latent-GMM; (iv) Gaussianity (skew/kurtosis, QQ) of latents vs raw features; (v) honest failure analysis if H1 doesn't help.
## Task C — GMM ablations (feed Krishna's master ablation)
Latent dim × cov-type × K grid → `ablation_gmm_*.csv` (val and test PR-AUC, 3 seeds).
## Task D — Theory (`p2_hybrid_theory.tex`)
Density estimation in a learned manifold; change of variables caveat (latent density ≠ data density — the AE decoder is not volume-preserving, so −log p(z) is a *score*, not a likelihood of x); curse of dimensionality via parameter count; why reconstruction error and latent density capture different anomaly types (off-manifold vs on-manifold-but-low-density); BIC on latents.
## Definition of Done
- [ ] Score files saved. [ ] Grid + AIC/BIC plots. [ ] Evidence (i)–(v). [ ] Ablation CSV. [ ] LaTeX. [ ] Viva notes. [ ] PR.
## 🎓 Viva
Why GMM on latents; what is and isn't a likelihood there; parameter-count arithmetic; whether H1 beat both parents and the CI that says so.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: at least 3 meaningful commits on at least 3 different calendar days (Phase 2).** Do not squash. Plan three sittings:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | hybrid_latent_gmm.py + first run on stub/real encodings |
| 2 | Day 2 | GMM grid on latents + scores + evidence (i)-(iii) |
| 3 | Day 3 | evidence (iv)-(v), ablation CSV, theory LaTeX, viva notes |
Each commit = a real, runnable increment, message format `[apoorva] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/apoorva-p2`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/apoorva-p2`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/apoorva.md` and tell Kushal.
