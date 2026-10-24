> **PHASE 2 = DEEP LEARNING + HYBRID.** Prerequisite: Phase 1 is merged and tagged `phase1-final`. Do not modify any Phase 1 file or result; build on top of them. Branch name: `feat/<name>-p2`.

# TASK 03 — Vani Rudra — Autoencoder (Deep Learning), Regularization & DL Literature
Project 26: Card Fraud Triage (GMM vs Autoencoder) — Team T033
**Rubric items you cover:** Phase 2 → *Architecture Logic*, *DL Dataset & Regularization*, *Technical Validation* (AE side), *Theoretical Rigor* (AE side).

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase2/P2_03_Vani_Autoencoder.md` completely. Implement ONLY the tasks in it. Only create/edit files under "Files you own". Never edit `src/data.py`, `src/metrics.py` or others' files; put change requests in `docs/requests/vani.md`. Never fabricate numbers or citations (mark uncertain refs `[VERIFY]`). The autoencoder is trained ONLY on legitimate training transactions; early stopping uses the legitimate-only validation loss; fraud labels never enter training. Seed everything. Branch `feat/vani-p2`, commits prefix `[vani]`. At the end write `docs/viva_notes/vani.md` explaining the architecture, loss, regularization, and every function in plain English.

## Prerequisite
Phase 1 merged & tagged `phase1-final`. Branch `feat/vani-p2`. `git pull origin main`. Start with `get_features("base")`. When Apoorva merges `engineered`, re-run the final model with `get_features("engineered")` (config flag) and report both.

## Files you own
```
src/autoencoder.py  src/train_ae.py
notebooks/04_autoencoder.ipynb
results/scores/ae_val.npy  ae_test.npy  ae_dae_val.npy  ae_dae_test.npy   # higher = more anomalous (= reconstruction MSE)
results/models/ae_*.pt   results/tables/ae_*.csv   results/figures/ae_*.png
reports/ieee_report/sections/p2_architecture.tex
reports/ieee_report/sections/p2_ae_results.tex
reports/refs_vani.bib
docs/viva_notes/vani.md
```

## Task A — Models (`src/autoencoder.py`, PyTorch)
1. `UndercompleteAE(d_in, latent_dim, hidden=[24,16], act='relu', dropout=0.0, batchnorm=False)` — symmetric encoder/decoder, linear output layer, `encode()` method exposing the bottleneck (Krishna's hybrid needs it).
2. `DenoisingAE` = same net, training adds Gaussian noise `x̃ = x + σ·ε` (σ ∈ {0.05,0.1,0.2,0.5}); loss compares reconstruction to the CLEAN x; no noise at inference.
3. Loss = MSE. Score = per-sample MSE (mean over features). Also compute per-feature error contribution for explanation.
4. Init: He/Kaiming for ReLU. Optimizer Adam (lr 1e-3, weight_decay 1e-5), batch 512, max 100 epochs, `ReduceLROnPlateau`, **early stopping patience 10 on val-LEGIT MSE only**. `torch.manual_seed`, deterministic flags, CPU/GPU auto.
5. Explain in notebook why the bottleneck must be *undercomplete* (otherwise identity mapping → everything reconstructs, fraud included).

## Task B — Training & experiments (`src/train_ae.py`, notebook 04)
- Run a documented hyper-parameter search (random or grid, ≤ 30 configs, budget stated): latent_dim ∈ {2,4,8,12,16}, hidden widths, dropout ∈ {0,0.1,0.2}, weight_decay ∈ {0,1e-5,1e-4}, σ for DAE, activation ∈ {relu, leaky_relu, tanh}. Selection criterion must be fixed BEFORE running: val PR-AUC (state that val labels are used for model selection only, never test). Also show val-legit reconstruction loss for transparency.
- Final: 5 seeds for the chosen undercomplete AE and DAE → mean ± std of PR-AUC and recall@FPR; save the median-seed scores as the official `ae_*.npy` / `ae_dae_*.npy`.
- Save learning curves (train vs val-legit loss), overlay early-stopping epoch.
- **Regularization study (rubric: DL Dataset & Regularization)** — table with val PR-AUC for: no regularization / +weight decay / +dropout / +input noise (DAE) / +early stopping / +batch-norm / +L1 sparsity on latent (optional sparse AE). Plot over/under-fit evidence (train–val gap).
- **DL dataset pipeline description**: tensors, `DataLoader` (shuffle only within train), normalization source (train legit), why no resampling/SMOTE (unsupervised, legit-only), class imbalance handled by design, memory/time per epoch.
- Evaluate through `src.metrics.evaluate_scores` → `results/tables/ae_metrics.csv`.
- Diagnostics: histogram of recon error legit vs fraud; top-20 false positives vs top-20 caught frauds and which features dominate error; latent space 2-D scatter (latent_dim=2 run, or PCA of codes) coloured by class; linear-AE ≈ PCA sanity check (compare to PCA reconstruction error with same k — coordinate with Krishna's PCA baseline).
- Technical validation: gradient-norm / loss-NaN checks, reconstruction of a held-out legit batch vs fraud batch, runtime, parameter count, inference latency per 10k transactions.

## Task C — (moved) DL literature review is owned by Prachee (`P2_01`). You must still know the key references for your viva.

## Task D — Theory (`p2_architecture.tex`)
Encoder f_θ, decoder g_φ, objective min E‖x−g(f(x))‖²; why a linear AE with MSE spans the PCA subspace; role of nonlinearity; denoising as learning the data manifold's score (Vincent 2011, `[VERIFY]`); bias–variance view of bottleneck size; why reconstruction error is an anomaly score and when it fails (anomalies in the principal subspace; scale dominated by high-variance features).

## Definition of Done
- [ ] 4 score files with correct lengths. [ ] Regularization ablation table. [ ] 5-seed mean±std. [ ] Learning curves.
- [ ] Viva notes. [ ] PR to `main`.

## 🎓 Human: what YOU must be able to say in your viva
Why undercomplete; why train only on legit; what early stopping monitors and why it doesn't leak; effect of σ and latent size; relation AE↔PCA; why MSE score can fail; every layer's shape; how many parameters; what regularizers you tested and what each did.

---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, rebase date rewriting, or history rewriting). Commits must carry the true time they were made.

**Goal: at least 3 meaningful commits, on at least 3 different calendar days.** Do not squash them. Plan your work in three sittings:
| Sitting | Day | Commit content (one or more commits) |
|---|---|---|
| 1 | Day 1 | src/autoencoder.py + basic training loop |
| 2 | Day 2 | hyper-param search, DAE, 5 seeds, scores |
| 3 | Day 3 | regularization study, DL review, LaTeX, viva notes |
Each commit must be a real, runnable increment with a clear message (`[vani] add ...`). Don't make empty or whitespace commits to hit the count.

**No-conflict rules**
1. Edit ONLY the files listed under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on your own branch `feat/vani`. Before every push run: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/vani`.
3. Open a PR to `main`; Kushal merges. Merge from the PR page (not force-push). Never `git push --force`.
4. Notebooks: only the owner edits their notebook; clear outputs of other people's notebooks never.
5. If you need a change in a shared file, write it to `docs/requests/vani.md` and tell Kushal.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: at least 3 meaningful commits on at least 3 different calendar days (Phase 2).** Do not squash. Plan three sittings:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | src/autoencoder.py + basic training loop + first scores |
| 2 | Day 2 | hyper-parameter search, DAE, 5 seeds, official scores |
| 3 | Day 3 | regularization study, theory LaTeX, viva notes |
Each commit = a real, runnable increment, message format `[vani] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/vani-p2`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/vani-p2`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/vani.md` and tell Kushal.
