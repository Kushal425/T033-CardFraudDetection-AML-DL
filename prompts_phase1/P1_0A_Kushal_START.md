> **PHASE 1 = MACHINE LEARNING ONLY.** Do not implement or mention neural networks / autoencoders / deep learning in code, report or notes (a one-sentence "future work" mention is allowed).

# TASK 00 — TEAM LEAD (START RUN) — Kushal Sarkar
**Run this FIRST. Everyone else is blocked until this is merged to `main`.**
Project 26: Card Fraud Triage — GMM Density Scoring vs Autoencoder Anomaly Detector (Team T033, Section A)

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase1/P1_0A_Kushal_START.md` completely. Implement ONLY the tasks in it. Create exactly the files listed under "Files you own". Do not invent data or numbers. If `data/raw/creditcard.csv` is missing, stop and tell me. Commit in small logical commits to branch `feat/lead-scaffold` with prefix `[lead]`. At the end, write `docs/viva_notes/kushal_start.md` explaining every function in plain English.

## Rules for the AI agent
1. Never fabricate results. Every number must come from code that was actually executed.
2. No leakage: scalers/models are fit on **training legitimate rows only**.
3. Fraud labels are NEVER used to fit any anomaly model. Validation labels may be used for model/hyperparameter selection and threshold-policy evaluation; validation legitimate scores are used to set FPR-budget thresholds. Test labels are used only for final evaluation, never for model or threshold selection.
4. Seed everything with `seed=42` from `configs/config.yaml`.
5. Add docstrings + comments so a student can explain each function in a viva.

## Human steps (Kushal, before running the agent)
- Download Kaggle "Credit Card Fraud Detection" (ULB, `mlg-ulb/creditcardfraud`) → place at `data/raw/creditcard.csv` (284,807 rows, 31 cols: `Time, V1..V28, Amount, Class`).
- Create the GitHub repo (private→share with all 5 + faculty), add all 4 teammates as collaborators, protect `main` (require PR).

## Files you own
```
README.md, requirements.txt, .gitignore, CONTRIBUTING.md, CONTRIBUTIONS.md, run_all.py
configs/config.yaml
src/__init__.py, src/config.py, src/data.py, src/utils.py, src/metrics.py
tests/test_no_leakage.py, tests/test_metrics.py
docs/data_card.md, docs/viva_notes/kushal_start.md
```
Also create EMPTY placeholder folders (with `.gitkeep`): `notebooks/ results/scores results/figures results/tables results/models reports/ieee_report/sections docs/viva_notes docs/requests docs/prompts data/processed`.
Copy the two prompt folders into `docs/prompts/phase1/` and `docs/prompts/phase2/` if not already there. Phase 1 execution must not require Phase 2 modules.

## Task A — `configs/config.yaml`
```yaml
seed: 42
paths: {raw: data/raw/creditcard.csv, processed: data/processed, scores: results/scores, figures: results/figures, tables: results/tables, models: results/models}
split: {train_frac: 0.60, val_frac: 0.20, test_frac: 0.20}   # chronological by Time, no shuffling
feature_set: base            # base | engineered (engineered is registered by Apoorva)
eval:
  fpr_budgets: [0.001, 0.005, 0.01]
  review_fractions: [0.001, 0.005, 0.01]
  top_k: [50, 100, 200]
```

## Task B — `src/data.py`
Implement:
- `load_raw()` — read CSV, assert schema & dtypes, raise a clear error if file missing. Report & log duplicate-row count (do not silently drop; keep a flag `drop_duplicates` in config default False).
- `chronological_split(df)` — sort by `Time` (stable), cut 60/20/20 by row order. NO shuffle. Save `data/processed/{train,val,test}.csv` (original columns, original index preserved) and `data/processed/split_manifest.json` containing: row counts, fraud counts, fraud rate (base rate), Time min/max per split.
- **Check**: if val or test contains < 50 frauds, print a loud warning and suggest 60/15/25; document the final decision in `docs/data_card.md`.
- `get_features(feature_set="base")` returning a dataclass `Splits`:
  `X_train_legit, X_train_all, y_train_all, X_val, y_val, X_test, y_test, feature_names, scaler`.
  - **base** feature set: V1..V28 (standardised), `log1p(Amount)` (standardised), `sin/cos(2π·(Time mod 86400)/86400)`; raw `Time` dropped. ⚠️ Document the ASSUMPTION that Time=0 ≈ start of day (unknown in the dataset).
  - StandardScaler is fit on `X_train_legit` only, then applied to all splits.
  - `X_train_all` exists ONLY for diagnostics; models must never fit on it.
  - A registry dict `FEATURE_SETS = {"base": _build_base}`; other members register more sets through `src/features.py` (do NOT import it if absent — use try/except).

## Task C — `src/metrics.py` (the single source of truth for scoring; everyone imports this)
Convention: **score higher = more anomalous (more likely fraud)**.
- `pr_auc(y, s)` → average precision. `roc_auc(y, s)`.
- `recall_at_fpr(y, s, fpr)` → highest recall such that FPR ≤ fpr (use ROC curve).
- `threshold_at_fpr(y_val, s_val, fpr)` → threshold chosen on validation legit scores (quantile 1−fpr).
- `precision_at_k(y, s, k)` and `precision_at_fraction(y, s, frac)` (review capacity).
- `recall_at_k`, `f1_at_threshold`, `confusion_at_threshold`.
- `evaluate_scores(y_val, s_val, y_test, s_test, cfg)` → one tidy `pandas.DataFrame` (rows = metrics; columns = val, test) where thresholds are chosen on val and APPLIED unchanged to test.
- `base_rate(y)`. Never report accuracy as the headline.

## Task D — `src/utils.py`
`set_seed`, `get_logger`, `save_scores(model_name, split, scores)` → `results/scores/{model_name}_{split}.npy`, `load_scores(...)`, `save_table(df, name)`, `save_fig(fig, name)`.
**Score-file contract** (everyone follows): 1-D float array, length = len(split), row order = the order of `data/processed/{split}.csv`, higher = more anomalous.

## Task E — Tests
- `test_no_leakage.py`: train max(Time) < val min(Time) < test min(Time); scaler mean equals train-legit mean; no fraud rows in `X_train_legit`.
- `test_metrics.py`: perfect scores → PR-AUC=1; random scores → PR-AUC≈base rate; recall_at_fpr monotone in fpr.

## Task F — Docs
- `README.md`: project title, one-paragraph problem, team table (name, enrollment, role), repo structure, how to reproduce (`pip install -r requirements.txt` → place data → `python run_all.py`), badges placeholder, "Not for production fraud decisions" disclaimer.
- `CONTRIBUTING.md`: branch per member `feat/<name>`, commit prefix `[name]`, PR into `main`, lead reviews, no force-push, no committing data/large files, commit **often** (individual contribution is evidence for viva).
- `CONTRIBUTIONS.md`: table of member → files → Phase 1 role → Phase 2 role (fill from the task files).
- `docs/data_card.md`: source, size, fraud rate, PCA-anonymised features (limits interpretability), duplicates, time span (~48h), split policy, known limitations.
- `run_all.py`: orchestrates available stages with try/except per stage and prints which stage ran. For the current Phase 1 run, include only the ML stages: eda, features, gmm, baselines, eval and final. Do not import or require ae, hybrid or ablation stages; those belong to Phase 2.

## Definition of Done
- [ ] `pytest` passes. [ ] `python -c "from src.data import get_features; s=get_features(); print(s.X_val.shape)"` works.
- [ ] `split_manifest.json` committed (manifest only, NOT the data). [ ] PR merged to `main`, tag `v0-scaffold`.
- [ ] Message the team: "Scaffold merged — pull main and run your task file."

---
## 🔁 Commit rule
Never set or alter commit dates or rewrite history. Commit the scaffold in several real, logical commits (config → data.py → metrics.py → tests → docs). Add to `CONTRIBUTING.md`: "Commit dates must be genuine; no history rewriting; rebase on `origin/main` before pushing; each member owns distinct files so PRs never conflict."
