> **PHASE 1 = MACHINE LEARNING ONLY.** Do not implement or mention neural networks / autoencoders / deep learning in code, report or notes (a one-sentence "future work" mention is allowed).

# PHASE 1 TASK 04 — Krishna Pramod Dubale — Evaluation Toolkit & Threshold Analysis
Project 26: Card Fraud Triage — Team T033
**Rubric items you cover:** *Model Application* (evaluation, thresholds), *Theoretical Rigor* (evaluation theory), *GitHub & Code Quality*.

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase1/P1_04_Krishna_Evaluation_Toolkit.md` completely. Implement ONLY the tasks in it. Only create/edit files under "Files you own". Never edit `src/data.py`, `src/metrics.py` or others' files; put change requests in `docs/requests/krishna.md`. Never fabricate numbers. Thresholds come from VAL only and are frozen before applying to TEST. Seed everything. Code against the score-file contract (`results/scores/{model}_{split}.npy`, higher = more anomalous); until Apoorva's and Vani's scores are merged, test with clearly-labelled synthetic stubs that are NEVER saved to `results/`. Branch `feat/krishna`, commits prefix `[krishna]`. No deep learning. At the end write `docs/viva_notes/krishna.md` explaining each method in plain English.

## Prerequisite
`git pull origin main`. Import metrics from `src.metrics`; build extras in `src/evaluation.py`.

## Files you own
```
src/evaluation.py
notebooks/05_evaluation.ipynb
results/tables/eval_*.csv   results/figures/eval_*.png
reports/ieee_report/sections/p1_evaluation_protocol.tex
reports/ieee_report/sections/p1_results_comparison.tex
reports/refs_krishna.bib
docs/viva_notes/krishna.md
```

## Task A — Toolkit (`src/evaluation.py`)
1. `bootstrap_ci(metric_fn, y, s, n=1000, seed)` — stratified bootstrap 95% CI for PR-AUC, ROC-AUC, recall@FPR, precision@capacity. Explain in the notebook why test CIs are wide (few frauds) and report fraud counts in the table.
2. `paired_bootstrap_diff(y, s_a, s_b, metric)` — CI of metric(A) − metric(B) on the same resamples; "significant" only if CI excludes 0.
3. **Threshold policies chosen on VAL, frozen, applied to TEST**: (a) FPR budget 0.1/0.5/1% (quantile of val-legit scores); (b) review capacity top-0.1/0.5/1% of transactions (and top-K 50/100/200); (c) max-F1 on val. Output per policy: threshold, #alerts, TP, FP, FN, precision, recall, F1, alerts per 10,000 transactions.
4. **Cost curve** with an assumed cost ratio (missed fraud = r × false alarm), sweep r ∈ {10,25,50,100,250,500}; state it is an ASSUMPTION; plot expected cost vs threshold and optimal threshold per r.
5. Plots (all models, same axes): PR curves with base-rate line, ROC (log-x), recall-vs-FPR, precision@K curve, score histograms legit vs fraud (log-y). Save with `save_fig`.
6. Calibration note: scores are not probabilities; optionally map rank→probability with Platt/isotonic fitted on VAL and show a reliability diagram with that caveat.
7. `compare_models(names)` → `results/tables/eval_master_phase1.csv`: PR-AUC (CI), ROC-AUC (CI), recall@FPR (CI), precision@capacity, train time, inference time, for GMM + Vani's baselines. Rank models and mark statistically indistinguishable ones (paired bootstrap vs the best).
8. Temporal-stability check: PR-AUC on val vs test per model (drift sensitivity), plus PR-AUC on 5 consecutive time-blocks of test if each has enough frauds (else say so).

## Task B — Theory (`p1_evaluation_protocol.tex`)
Why accuracy fails (0.17% base rate); PR vs ROC under imbalance; average precision definition; recall@FPR from the ROC curve; precision at review capacity (operational view); bootstrap CI procedure and its assumptions; why thresholds are set on val; limitations (few frauds, assumed costs).

## Task C — Results LaTeX (`p1_results_comparison.tex`)
Master table, PR-curve figure, threshold table, cost-curve figure, 1-paragraph reading of each. Numbers pulled from CSV, not typed.

## Definition of Done
- [ ] Toolkit unit-tested with stubs. [ ] Master table with CIs once scores merged. [ ] Threshold + cost tables. [ ] All plots. [ ] LaTeX. [ ] Viva notes. [ ] PR to `main`.

## 🎓 Human: what YOU must be able to say in your viva
Bootstrap procedure step by step; why PR-AUC; recall@FPR=0.5% in plain words; precision at review capacity; why thresholds from val; what the cost curve assumes; whether differences between models are significant and how you know; why test results have wide CIs.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: make meaningful incremental commits as work is completed (Phase 1).** Do not squash useful history or manipulate dates. Plan the work in the suggested sittings, adapting to the real deadline:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | metrics extensions + bootstrap CI + stub tests |
| 2 | Day 2 | threshold policies + cost curve + plots |
| 3 | Day 3 | master table with real scores, LaTeX, viva notes |
Each commit should be a real, runnable increment, message format `[krishna] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/krishna`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/krishna`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/krishna.md` and tell Kushal.
