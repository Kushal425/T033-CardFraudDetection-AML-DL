# FINAL_AUDIT.md — Integration Audit Log
**Author:** Kushal Sarkar (Lead) | **Task:** P1_05 Final Run | **Date:** 2026-10-25

---

## Step 1 — PASS/FAIL Table

| Check | Status | Notes |
|---|---|---|
| Score files exist (14 files: 7 models × val/test) | ✅ PASS | All 14 `.npy` files present |
| Score lengths match split sizes (val=56961, test=56962) | ✅ PASS | All verified |
| No NaN or Inf in any score file | ✅ PASS | Confirmed by `test_score_contract.py` |
| Real models PR-AUC > base_rate (test) | ✅ PASS | All 5 real models beat 0.0013 |
| Random reference ≈ base_rate | ✅ PASS | random PR-AUC=0.0015 ≈ base rate |
| `pytest` passes (all 80 tests) | ✅ PASS | 80 passed (40 original + 40 score contract) |
| No test leakage (hyper-param selection on val only) | ✅ PASS | grep confirms no test labels used for tuning |
| `data/raw/creditcard.csv` not in git | ✅ PASS | `.gitignore` confirmed |
| No `.DS_Store` or model binaries in git | ⚠️ FLAG | `.DS_Store` present on local; remove if staged |
| Notebooks run top-to-bottom | ✅ PASS | All 5 notebooks executed with outputs |
| LaTeX section files exist | ✅ PASS | 9 `.tex` section files present |
| Score convention (fraud mean > legit mean) | ✅ PASS | All 5 real models verified by test |

---

## Step 2 — Commit Author Audit

| Author | Commits | Notes |
|---|---|---|
| Kushal425 / Kushal Sarkar | 7 | Lead: scaffold + final |
| Apoorv Choudhary / Apoorva choudhary | 6 | Slight name inconsistency (same person) |
| pracheedhar | 4 | Prachee Dhar |
| vanosaur / Vani Rudra | 5 | Slight name inconsistency (same person) |
| Krishna Pramod Dubale / KrishnaDubale | 4 | Slight name inconsistency (same person) |

> ⚠️ **Name inconsistency:** Git records slightly different author names for Apoorva, Vani, and Krishna (local vs GitHub config difference). This does not affect contribution validity but should be noted. All 5 members have genuine commits with meaningful content.

---

## Step 3 — Score Summary (from actual `.npy` files)

| Model | PR-AUC (val) | PR-AUC (test) | Recall@FPR=0.5% (test) | Prec@0.5%cap (test) |
|---|---|---|---|---|
| **PCA Recon.** | **0.7066** | **0.7425** | **0.7733** | 0.2000 |
| LOF | 0.4069 | 0.4478 | 0.7200 | 0.1860 |
| GMM | 0.1950 | 0.4142 | 0.6933 | 0.1825 |
| OC-SVM | 0.0613 | 0.1322 | 0.6933 | 0.1649 |
| Iso. Forest | 0.0330 | 0.0577 | 0.3467 | 0.0772 |
| Random (ref) | 0.0011 | 0.0015 | 0.0133 | 0.0035 |
| Amount (ref) | 0.0033 | 0.0034 | 0.0133 | 0.0035 |

**Winner: PCA Reconstruction Error** dominates on PR-AUC and recall@FPR. GMM underperforms in val (0.195) but improves substantially on test (0.414) — flagged below.

---

## Step 4 — Flagged Items (Non-Blocking for Phase 1)

1. **GMM val/test PR-AUC gap (0.195 → 0.414):** Unusual reversal. GMM likely selected sub-optimal K on the small val set (57 frauds → unstable PR-AUC). The test improvement may reflect that test's fraud distribution better aligns with the GMM's learned density. Documented here; no change to scientific choices.

2. **Isolation Forest low PR-AUC (0.058):** Significantly underperforms. Likely due to the high dimensionality (31 features) and the fact that fraud transactions are not well-isolated by random partitioning in this dataset. Documented in results discussion.

3. **OC-SVM recall@FPR=0.1% = 0.0:** The decision boundary at this very tight FPR constraint flags zero frauds. At FPR=0.5%, performance improves (0.693). Operational note: OC-SVM requires a looser budget.

4. **Val has only 57 frauds:** Wide confidence intervals on all val-based metrics. Bootstrap CIs from Krishna's evaluation toolkit should be cited in the report.

5. **Seaborn not in requirements.txt:** Added before tests. `requirements.txt` updated.

---

## Step 5 — No Fabricated Numbers

All numbers in `results/tables/final_comparison.csv` and `results/tables/paper_numbers.tex` are computed directly from `results/scores/*.npy` by `src/final.py`. No numbers were manually typed.
