> **PHASE 2 = DEEP LEARNING + HYBRID.** Prerequisite: Phase 1 is merged and tagged `phase1-final`. Do not modify any Phase 1 file or result; build on top of them. Branch name: `feat/<name>-p2`.

# PHASE 2 TASK 01 — Prachee Dhar — DL Literature Review, DL Data Pipeline Documentation & Phase 2 Report Sections
**Rubric items you cover:** *Deep Learning Literature Review*, *DL Dataset & Regularization* (documentation & data-side analysis), *Presentation*.

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase2/P2_01_Prachee_DL_Literature_Pipeline.md` completely. Implement ONLY the tasks in it. Only create/edit files under "Files you own". Never edit shared files or others' files; use `docs/requests/prachee.md`. Never fabricate citations (mark each `[VERIFY]`) or numbers (read CSV/code outputs). Branch `feat/prachee-p2`, commits prefix `[prachee]`. Never alter commit dates. At the end write `docs/viva_notes/prachee_p2.md`.

## Files you own
```
reports/dl_literature_review.md   reports/refs_prachee_p2.bib
reports/ieee_report/sections/p2_introduction.tex
reports/ieee_report/sections/p2_dl_literature.tex
reports/ieee_report/sections/p2_dl_data_pipeline.tex
notebooks/07_dl_data_checks.ipynb   src/eda_dl.py
results/tables/dl_data_*.csv   results/figures/dl_data_*.png
docs/slides_phase2_data_section.md
docs/viva_notes/prachee_p2.md
```
## Task A — DL literature review (`reports/dl_literature_review.md`, ≥ 1.5 pages, ≥ 10 refs, all `[VERIFY]`)
Autoencoders (Hinton & Salakhutdinov 2006); denoising AE (Vincent et al. 2008); AE anomaly detection (Sakurada & Yairi 2014; An & Cho 2015); DAGMM (Zong et al. 2018); Adam (Kingma & Ba 2015); dropout (Srivastava et al. 2014); batch norm (Ioffe & Szegedy 2015); *Deep Learning* (Goodfellow et al. 2016) for regularization/early stopping; over-generalisation of AEs to anomalies; fraud-specific deep learning work. Comparison table (Paper | Year | Method | Dataset | Metric | Limitation) + explicit gap statement tying to our design (legit-only training, chronological split, val-chosen thresholds, hybrid with a justified rationale).
## Task B — DL data pipeline checks (`src/eda_dl.py`, notebook 07)
Using Vani's tensors/DataLoader code (read-only): verify shapes, dtype, no NaN, train-legit normalisation stats equal Phase 1 scaler, batch composition, no fraud in training batches (assert), val-legit used for early stopping, memory/epoch time; distribution of reconstruction-relevant statistics (feature variances — high-variance features dominate MSE: show it); effect of train-only normalisation; document why no oversampling/SMOTE.
## Task C — LaTeX
`p2_introduction.tex` (recap Phase 1 in 1 paragraph + Phase 2 motivation/contributions), `p2_dl_literature.tex`, `p2_dl_data_pipeline.tex`.
## Task D — Slides content `docs/slides_phase2_data_section.md` (3–4 slides: DL literature gap, data pipeline, leakage controls).
## Definition of Done
- [ ] ≥ 10 `[VERIFY]` refs. [ ] Pipeline assertions pass. [ ] LaTeX. [ ] Viva notes. [ ] PR to `main`.
## 🎓 Viva
Why legit-only training; why val-legit early stopping doesn't leak; why high-variance features dominate MSE; what each reference contributes; the gap.


---
## 🔁 Commit cadence & conflict-free pushing (REAL dates only)
**Hard rule for the AI agent:** never set or alter commit dates (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date`, no history rewriting). Commits carry the true time they were made.

**Goal: at least 3 meaningful commits on at least 3 different calendar days (Phase 2).** Do not squash. Plan three sittings:
| Sitting | Day | Commit content |
|---|---|---|
| 1 | Day 1 | DL literature review draft + refs |
| 2 | Day 2 | src/eda_dl.py + notebook 07 checks |
| 3 | Day 3 | LaTeX sections + slides content + viva notes |
Each commit = a real, runnable increment, message format `[prachee] add ...`. No empty/whitespace commits.

**No-conflict rules**
1. Edit ONLY the files under "Files you own". Never touch shared files (`src/data.py`, `src/metrics.py`, `src/utils.py`, `configs/config.yaml`, README).
2. Work on branch `feat/prachee-p2`. Before every push: `git fetch origin && git rebase origin/main`, then `git push -u origin feat/prachee-p2`.
3. Open a PR to `main`; Kushal merges from the PR page. Never `git push --force`.
4. Only the owner edits their notebook.
5. Need a shared-file change? Write it to `docs/requests/prachee.md` and tell Kushal.
