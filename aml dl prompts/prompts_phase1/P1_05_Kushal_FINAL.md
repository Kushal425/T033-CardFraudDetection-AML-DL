> **PHASE 1 = MACHINE LEARNING ONLY.** Do not implement or mention neural networks / autoencoders / deep learning in code, report or notes (a one-sentence "future work" mention is allowed).

# PHASE 1 TASK 05 — TEAM LEAD (FINAL RUN) — Kushal Sarkar
**Run ONLY after Prachee, Apoorva, Vani and Krishna have merged their PRs into `main`.**

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase1/P1_05_Kushal_FINAL.md` completely. First run the integration audit and print a PASS/FAIL table. Fix only integration problems (paths, imports, contract violations); never alter teammates' scientific choices — log suspicious items in `docs/FINAL_AUDIT.md`. Never fabricate numbers: all report/notebook numbers are read from `results/tables/*.csv`. Never alter commit dates or rewrite history. Branch `feat/lead-final`, prefix `[lead]`. No deep learning. At the end write `docs/viva_notes/kushal_final.md`.

## Step 1 — Integration audit
- [ ] `git log --format='%an' | sort | uniq -c` and distinct commit days per author (`git log --format='%an %ad' --date=short | sort -u | cut -d' ' -f1 | uniq -c`) → every member ≥ 3 commits on ≥ 3 days; if not, flag in `docs/FINAL_AUDIT.md` (fix only with genuine remaining work).
- [ ] Score files exist: gmm, pca, iforest, ocsvm, lof, random, amount × {val,test}; lengths = split sizes; no NaN/inf; PR-AUC > base rate for real models, ≈ base rate for random.
- [ ] `pytest` passes; add `tests/test_score_contract.py`.
- [ ] No test leakage: hyper-parameter selection code reads only val metrics (grep); test appears only in final tables.
- [ ] Fresh clone → new venv → `pip install -r requirements.txt` → `python run_all.py` works (or document runtime/cached stages).
- [ ] No data/model binaries, secrets or `.DS_Store` committed.
- [ ] Notebooks 01, 02, 03, 03b, 05, 06 run top-to-bottom.
- [ ] LaTeX figures exist; merge the four `refs_*.bib` into `reports/ieee_report/refs.bib`, dedupe, remove `[VERIFY]` only after manual verification.

## Step 2 — Final comparison (`notebooks/06_final_comparison.ipynb`, `src/final.py`)
Headline table (GMM vs PCA vs Isolation Forest vs OC-SVM vs LOF) with CIs; answer: *Which is best at fixed FPR budgets and review capacity? Does GMM beat simpler methods significantly? Where does each fail? What is the false-alert burden per 10k transactions?* Error-analysis gallery, threats to validity (few test frauds, PCA features, 2-day window, single dataset, assumed costs), disclaimer (research prototype). Write `results/tables/paper_numbers.tex` with `\newcommand` numbers.

## Step 3 — Report (IEEE, `reports/ieee_report/main_phase1.tex`, `\documentclass[conference]{IEEEtran}`)
Abstract · Introduction · Related Work · Dataset & EDA · Feature Engineering · GMM Theory & Results · Baselines · Evaluation Protocol & Results · Discussion & Limitations (you) · Conclusion & Future Work (autoencoder/hybrid = next phase, one sentence) · Individual Contributions · References. 6–8 pages, compiles with `latexmk -pdf`, no undefined refs, every figure/table cited in text.

## Step 4 — Slides outline `docs/slides_phase1_outline.md`
12–15 slides, one message each, ≤ 3 min per member: problem → data/EDA → features → GMM → baselines → evaluation → results → demo → limitations → contributions → backup (EM derivation, AIC/BIC plot, threshold table, CI method).

## Step 5 — Repo polish & tag
README with headline table + reproduction steps + disclaimer + license; `CONTRIBUTIONS.md` with real commit counts; tag `phase1-final`; GitHub Release with the PDF.

## Definition of Done
- [ ] Audit PASS (or documented). [ ] Report PDF. [ ] Slides outline. [ ] Fresh clone reproduces. [ ] Tag pushed. [ ] Mock viva done (`guides/VIVA_GUIDE.md`).

## Commit rule
Never alter commit dates or rewrite history.
