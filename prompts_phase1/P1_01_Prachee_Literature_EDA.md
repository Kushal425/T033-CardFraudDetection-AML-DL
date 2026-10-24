> **PHASE 1 = MACHINE LEARNING ONLY.** Do not implement or mention neural networks / autoencoders / deep learning in code, report or notes (a one-sentence "future work" mention is allowed).

# TASK 01 — Prachee Dhar — Literature Review, EDA & Dataset Quality
Project 26: Card Fraud Triage (GMM vs Autoencoder) — Team T033
**Rubric items you cover:** Phase 1 → *Literature Review*, *Dataset Quality & EDA*

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase1/P1_01_Prachee_Literature_EDA.md` completely. Implement ONLY the tasks in it. Only create/edit the files under "Files you own". Never edit `src/data.py`, `src/metrics.py` or other members' files — if you need a change, write it in `docs/requests/prachee.md`. Never fabricate citations: only cite papers you are certain exist, and mark each with `[VERIFY]` so I can confirm it. Never fabricate numbers: every statistic comes from executed code. Work on branch `feat/prachee`, commit often with prefix `[prachee]`. At the end write `docs/viva_notes/prachee.md` explaining every plot and every decision in plain English.

## Prerequisite
`git pull origin main` (needs Kushal's scaffold: `src/data.py`, `configs/config.yaml`). Use `from src.data import load_raw, get_features` and the saved splits only.

## Files you own
```
notebooks/01_literature_and_eda.ipynb
src/eda.py
reports/literature_review.md
reports/ieee_report/sections/p1_introduction.tex
reports/ieee_report/sections/p1_related_work.tex
reports/ieee_report/sections/p1_dataset_eda.tex
results/figures/eda_*.png   results/tables/eda_*.csv
docs/viva_notes/prachee.md
reports/refs_prachee.bib
```

## Task A — Literature review (`reports/literature_review.md`, ≥ 2 pages)
Structured as: (1) Problem & why fraud is anomaly detection (extreme imbalance, label scarcity, concept drift); (2) Density-based methods — **GMM & EM** (Dempster, Laird, Rubin 1977), model selection AIC/BIC (Akaike 1974; Schwarz 1978); (3) Briefly state the project scope and its limits; keep any later-phase method to one sentence in future work and do not review or evaluate it in the Phase 1 report; (4) Classical anomaly baselines (Isolation Forest — Liu et al. 2008; One-class SVM — Schölkopf et al. 2001; PCA reconstruction); (5) The credit-card dataset itself (Dal Pozzolo et al. 2015 and related ULB papers); (6) Evaluation on imbalanced data (Saito & Rehmsmeyer 2015 PR vs ROC; Davis & Goadrich 2006); (7) **Gap statement**: what existing work does not do and how our project fills it (fair same-train/same-test comparison, chronological split, threshold selected on val, review-capacity metrics).
- Include a comparison table: Paper | Year | Method | Dataset | Metric | Limitation.
- Mark every reference `[VERIFY]` until Kushal confirms it. ≥ 12 verified references (keep `[VERIFY]` markers only during drafting; remove them after checking each source). Also produce `reports/refs_prachee.bib`.

## Task B — Dataset quality (`src/eda.py` + notebook)
Compute and save tables/figures for:
1. Shape, dtypes, memory, missing values (expected 0 — verify), **duplicate rows** count & whether duplicates straddle classes.
2. Class imbalance: counts, fraud rate (≈0.172%), bar + log-scale plot. Per-split base rates from `split_manifest.json`.
3. `Amount`: distribution (raw & log1p), fraud vs legit comparison, outliers (IQR), zero-amount transactions.
4. `Time`: span (~172,792 s ≈ 48 h), transactions per hour, fraud per hour, **drift check**: compare train vs val vs test distributions of each feature (KS-test statistic per feature + bar chart; Population Stability Index optional).
5. V1–V28: fraud-vs-legit density plots (KDE) for all 28 (grid); rank features by KS statistic / Cohen's d between classes; correlation heatmap (legit only) — note V's are PCA components so near-zero correlation is EXPECTED; explain.
6. Multivariate view: 2-D PCA/t-SNE (subsample 10k legit + all fraud, seeded) coloured by class — comment honestly on overlap.
7. Gaussianity check on training-legit features (QQ-plots for 4 features, skew & kurtosis table) — this is key evidence for/against the **GMM assumption**; hand findings to Apoorva via `results/tables/eda_gaussianity.csv`.
8. Leakage audit table: statement per rule (scaler fit on train legit only; labels unused in fitting; chronological split; duplicates).
9. **Data limitations** paragraph: PCA-anonymised (no domain interpretation), only 2 days (no long-term drift, no seasonality), European cardholders Sept 2013, strong class imbalance, label noise unknown.

Every figure: titled, axes labelled, saved by `save_fig`. Every plot followed by a markdown cell "What this tells us / how it affects modelling" (1–3 sentences).

## Task C — LaTeX sections (IEEEtran-compatible, plain `\section{}` content only, no preamble)
`p1_introduction.tex` (motivation, problem statement, contributions, paper outline), `p1_related_work.tex` (from Task A), `p1_dataset_eda.tex` (dataset description + 3–4 key figures + data limitations). Use `\cite{}` keys matching the .bib.

## Definition of Done
- [ ] Notebook runs top-to-bottom with `Restart & Run All` and no errors. [ ] ≥ 12 `[VERIFY]` refs. [ ] Gaussianity CSV exists.
- [ ] Viva notes written. [ ] PR opened to `main`, assigned to Kushal.

## 🎓 Human: what YOU must be able to say in your viva
Why PR-AUC not accuracy; what the 0.172% base rate means; why PCA features limit interpretation; what drift you found; why EDA says GMM is/isn't reasonable; what each reference contributes.

---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, rebase date rewriting, or history rewriting). Commits must carry the true time they were made.

**Goal: make multiple meaningful commits as work is completed.** Do not create empty commits or manipulate dates; use the suggested sittings as a planning aid, not a fixed calendar-day requirement:
| Sitting | Day | Commit content (one or more commits) |
|---|---|---|
| 1 | Day 1 | load data + first EDA tables/figures |
| 2 | Day 2 | literature_review.md + remaining EDA + gaussianity table |
| 3 | Day 3 | LaTeX sections + viva notes + polish |
Each commit must be a real, runnable increment with a clear message (`[prachee] add ...`). Don't make empty or whitespace commits to hit the count.

**No-conflict rules**
1. Edit ONLY the files listed under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on your own branch `feat/prachee`. Before every push run: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/prachee`.
3. Open a PR to `main`; Kushal merges. Merge from the PR page (not force-push). Never `git push --force`.
4. Notebooks: only the owner edits their notebook; clear outputs of other people's notebooks never.
5. If you need a change in a shared file, write it to `docs/requests/prachee.md` and tell Kushal.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: make meaningful incremental commits as work is completed (Phase 1).** Do not squash useful history or manipulate dates. Plan the work in the suggested sittings, adapting to the real deadline:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | load data + first EDA tables/figures |
| 2 | Day 2 | literature_review.md + remaining EDA + Gaussianity table |
| 3 | Day 3 | LaTeX sections + viva notes + polish |
Each commit should be a real, runnable increment, message format `[prachee] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/prachee`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/prachee`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/prachee.md` and tell Kushal.
