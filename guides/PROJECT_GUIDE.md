# PROJECT GUIDE — Team T033, Section A — Project 26

> **Current work is Phase 1 only.** Use `PHASE1_ONLY.md` and `prompts_phase1/` as the source of truth for current tasks. Phase 2 descriptions below are retained for future reference, not current submission requirements.
**Card-Fraud Triage: GMM Density Scoring vs Autoencoder Anomaly Detector**
Team: Kushal Sarkar (lead, 2401020033) · Prachee Dhar (2401010330) · Apoorva Choudhary (2401010092) · Vani Rudra (2401010490) · Krishna Pramod Dubale (2401010239)
Faculty: Shobhit Johri · Arun Chauhan · Ashwin Tewary (@nst.rishihood.edu.in)

---
## 1. What the problem statement requires (from the faculty PDF)
| Item | Requirement |
|---|---|
| Idea | Frame card fraud as anomaly detection; fit models **primarily on legitimate transactions**; compare **probability density scores (GMM)** vs **reconstruction error (autoencoder)** on a held-out **mixed** set |
| AML baseline | GMM on legit transactions; **tune K with AIC and BIC**; low log-likelihood = anomaly score |
| Gap | GMM assumes Gaussian-mixture manifold; unstable in high dimensions |
| DL model | Undercomplete or **denoising** autoencoder trained **only on legit training** transactions; reconstruction error = score |
| Fair comparison | Same legit-only training set + untouched mixed test set. Report **PR-AUC, recall at fixed FPR budgets, precision at review capacity. Do NOT rely on accuracy** |
| Setup tips | Chronological split if possible; scaling + GMM fit on legit train rows only; thresholds from validation; fraud label hidden during unsupervised training; report base rates |
| Deliverable | Fraud triage notebook with GMM and AE precision-recall curves, threshold analysis, documented false-positive trade-offs |
| Dataset | Kaggle/ULB Credit Card Fraud Detection (284,807 tx, 492 frauds, 0.172%). You may swap for a dataset at least as complex — not needed, don't risk it |

## 2. Evaluation structure (both after mid-sems; dates to be announced)
**Phase 1 — ML only:** Literature Review · Dataset Quality & EDA · Feature Engineering · Theoretical Rigor · Model Application · GitHub Repo & Code Quality · Presentation · Viva Voce · Report (LaTeX/IEEE)
**Phase 2 — DL (+ Hybrid for bonus):** Architecture Logic · Hybrid Innovation · Ablation Studies · DL Literature Review · DL Dataset & Regularization · Technical Validation · Theoretical Rigor · GitHub Repo & Code Quality · Presentation · Viva Voce · Report (LaTeX/IEEE)
**Hybrid bonus:** extra marks only if hybrid is *meaningful, technically justified, successfully implemented, with demonstrated value*. Mere ML+DL stacking with no rationale = no marks. (Our H1/H2/H3 + overlap analysis + ablation are designed for this.)
**Viva:** INDIVIDUAL, even though project is team-based. Each person must explain methods, implementation, experiments, results, theory.

## 3. What to SUBMIT (checklist)
> The official instructions do not state an exact upload portal or final submission deadline. Confirm the submission channel, deadline and any file/page/slide limits with faculty. Prepare the Phase 1 artifacts listed below; Phase 2 items are future reference only.

**Current Phase 1 pack**
- [ ] GitHub repo link and reviewed `phase1-final` tag
- [ ] Phase 1 report PDF + LaTeX source (IEEE format)
- [ ] Phase 1 presentation (PPT/PDF)
- [ ] Relevant Phase 1 notebooks executed with outputs (EDA, GMM, classical baselines, evaluation, final comparison)
- [ ] README reproduction steps
**Phase 2 pack**
- [ ] Same repo, tag `phase2-final`
- [ ] Phase-2 Report PDF + LaTeX source (IEEE)
- [ ] Presentation
- [ ] Notebooks 04–06, ablation tables, hybrid analysis
- [ ] Individual understanding evidenced in `docs/viva_notes/` + contribution matrix

## 4. Execution plan (who runs what, in what order)
| Order | Who | File | Gate |
|---|---|---|---|
| 1 | Kushal | `00_LEAD_START` | merge scaffold → tag `v0-scaffold` |
| 2 (parallel) | Prachee | `01` | EDA + lit review |
| 2 | Apoorva | `02` | GMM scores |
| 2 | Vani | `prompts_phase1/P1_03_Vani_Classical_Baselines.md` | Classical baseline scores (Phase 1) |
| 2 | Krishna | `prompts_phase1/P1_04_Krishna_Evaluation_Toolkit.md` | Evaluation toolkit and comparisons (Phase 1) |
| 3 | Kushal | `05_LEAD_FINAL` | audit → final report → tags |
Phase 1 deliverables (Prachee, Apoorva, Krishna-Part-1) should be frozen & tagged BEFORE Phase 1 evaluation; Phase 2 work (Vani, Krishna-Part-2) completes before Phase 2.

**How each teammate uses their file:** `git pull` → `git checkout -b feat/<name>` → open repo in Antigravity → paste the prompt at the top of their md → review the output → run notebooks themselves → commit/push → open PR.

## 5. ⚠️ Honest cautions (read these)
1. **Understanding > code.** AI-generated code is fine, but the viva is individual. Each member must read their own code and `docs/viva_notes/<name>.md`, re-run their notebook, and be able to modify a parameter live. A member who can't explain their part will lose viva marks regardless of repo quality.
2. **Antigravity/AI hallucination risk:** verify every citation (`[VERIFY]` tags), every number, and every claim of "state-of-the-art". Wrong or invented references are worse than few references.
3. **Test set is sacred.** Choose K, architecture, σ, fusion weight, thresholds on val only. If someone peeks at test to tune, the whole comparison is invalid and you'd be exposed in viva.
4. **Few frauds in validation/test** → wide CIs and unstable threshold estimates. Report actual fraud counts and uncertainty; do not over-claim model superiority unless paired comparisons support it.
5. **Features are PCA-anonymised** → FE and interpretability are limited; say so openly and show principled choices.
6. Commit history is evidence of contribution: commit genuine, meaningful work as it is completed. Never fabricate commits or manipulate commit dates.
7. Don't copy another team's code; similar-looking projects exist in the section (same ideas PDF). Your hybrid + ablation + error analysis is your differentiator.

## 6. FINAL CHECK — mapped to the rubric (target = full marks)
### Phase 1
| Criterion | Evidence you must have | ✔ |
|---|---|---|
| Literature Review | ≥12 verified refs; comparison table; explicit gap | ☐ |
| Dataset Quality & EDA | duplicates, imbalance, drift (KS), Amount/Time, Gaussianity check, leakage audit, limitations | ☐ |
| Feature Engineering | feature-set comparison table, justified choice, no leakage | ☐ |
| Theoretical Rigor | EM derivation, AIC/BIC, param count, singularity, density≠fraud | ☐ |
| Model Application | GMM grid (cov type × K), AIC/BIC plots, PR/ROC, FPR/capacity metrics, thresholds from val, classical baselines, CIs | ☐ |
| GitHub & Code Quality | README, structure, tests, requirements, seeds, small commits from all 5, no data in repo | ☐ |
| Presentation | 12–15 slides, each member speaks, demo, limitations | ☐ |
| Viva | everyone passes mock viva | ☐ |
| Report | IEEE LaTeX, compiles, numbers from CSV, figures referenced | ☐ |
### Phase 2
| Criterion | Evidence | ✔ |
|---|---|---|
| Architecture Logic | why undercomplete/denoising, layer sizes, loss, score, link to PCA | ☐ |
| Hybrid Innovation | H1/H2/H3 with rationale, value shown vs parents, overlap analysis | ☐ |
| Ablation Studies | table in Krishna's file, one-factor-at-a-time, seeds, CIs | ☐ |
| DL Literature Review | ≥8 verified refs, gap | ☐ |
| DL Dataset & Regularization | pipeline description, regularization table, train/val curves | ☐ |
| Technical Validation | 5 seeds mean±std, paired bootstrap, sanity checks, runtime/params | ☐ |
| Theoretical Rigor | AE objective, denoising theory, failure modes | ☐ |
| Repo / Presentation / Viva / Report | as above, IEEE Phase-2 report | ☐ |

## 7. Your own (lead) to-do list in plain words
**Before running anything:** download dataset → create GitHub repo → invite team → protect `main`.
**Run 1 (`00`)**: scaffold + shared code. **While team works:** review PRs within 24h, keep `main` green, answer `docs/requests/*.md`.
**Run 2 (`P1_05`) for the current submission**: audit Phase 1 only → final comparison → Phase 1 report → Phase 1 slides outline → tag `phase1-final`. Do not assemble a Phase 2 report now.
**Also yours:** email faculty (single point of contact), confirm submission format & dates, schedule 2 mock vivas (see `VIVA_GUIDE.md`), final proofread of both PDFs.
