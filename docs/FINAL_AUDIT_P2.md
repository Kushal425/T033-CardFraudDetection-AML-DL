# FINAL_AUDIT_P2.md — Phase 2 Integration Audit Log
**Author:** Kushal Sarkar (Phase 2 Integration Lead)  
**Task:** P2_05 Final Integration, Research Validation & Report  
**Date:** 2026-11-16  
**Branch:** `feat/lead-final-p2`  
**Base Commit:** `f7d92c8` (latest `origin/main`)

---

## 1. Executive Summary & Overall Audit Verdict

| Overall Audit Verdict | Status | Notes |
|---|---|---|
| **Phase 2 Integration Status** | **✅ PASS** | All 4 teammate PRs (#13 Krishna, #14 Vani, #15 Apoorva, #16 Prachee) merged into `main` |
| **Score File Integrity** | **✅ PASS** | All 20 score files ($10 \text{ models} \times 2 \text{ splits}$) exist, match length, zero NaN/Inf, follow convention |
| **Anti-Leakage Invariants** | **✅ PASS** | Legit-only AE training ($N=170,524$), val-legit early stopping, val-frozen fusion & thresholds |
| **Automated Test Suite** | **✅ PASS** | **139 / 139 passing tests** across unit, invariant, and contract suites |
| **Scientific Integrity** | **✅ PASS** | Zero fabricated numbers; all metrics derived directly from empirical score files |

---

## 2. PASS / FAIL Integration Checklist

| Verification Category | Requirement | Status | Evidence & Audit Verification |
|---|---|---|---|
| **Branch & Git State** | Clean branch `feat/lead-final-p2` based on `origin/main` | ✅ PASS | `git switch -c feat/lead-final-p2 origin/main` |
| **Merged Teammate PRs** | PR #13 (Krishna - DL Lit & Data Pipeline) | ✅ PASS | Commit `5a88be2` merged into `main` |
| | PR #14 (Vani - Deep Autoencoders & DAE) | ✅ PASS | Commit `fc22021` merged into `main` |
| | PR #15 (Apoorva - Latent-Space GMM & Theory) | ✅ PASS | Commit `05863d3` merged into `main` |
| | PR #16 (Prachee - Hybrids, Ablation & Master Eval) | ✅ PASS | Commit `f7d92c8` merged into `main` |
| **Score Files (DL & Hybrid)** | `ae_{val,test}.npy` exists, shape 1-D | ✅ PASS | Lengths: val=56,961, test=56,962; no NaN/Inf |
| | `ae_dae_{val,test}.npy` exists, shape 1-D | ✅ PASS | Lengths: val=56,961, test=56,962; no NaN/Inf |
| | `hybrid_latentgmm_{val,test}.npy` exists, shape 1-D | ✅ PASS | Lengths: val=56,961, test=56,962; no NaN/Inf |
| | `hybrid_fusion_{val,test}.npy` exists, shape 1-D | ✅ PASS | Lengths: val=56,961, test=56,962; no NaN/Inf |
| | `hybrid_dagmmlite_{val,test}.npy` exists, shape 1-D | ✅ PASS | Lengths: val=56,961, test=56,962; no NaN/Inf |
| **Score Direction Convention** | Higher score = more anomalous ($\text{mean}_{\text{fraud}} > \text{mean}_{\text{legit}}$) | ✅ PASS | Verified across all 10 models for both validation and test sets |
| **Legitimate-Only Training** | DL models trained strictly on legitimate records | ✅ PASS | Asserted in `test_dl_data.py` & `test_score_contract_phase2.py`: 0 frauds in train tensors ($N=170,524$) |
| **Anti-Leakage Controls** | Validation-only model selection & threshold freezing | ✅ PASS | Validation-legit early stopping; H2 grid search selects $w=0.0$ on val; thresholds frozen on val |
| **Test Set Integrity** | Held-out test evaluated exactly once for final reporting | ✅ PASS | No test labels used during tuning or parameter selection |
| **Test Suite** | Full automated CI passing | ✅ PASS | **139 tests passed** (`pytest tests/`) |
| **Phase 2 Notebooks** | Executable notebooks in `notebooks/` | ✅ PASS | `04_autoencoder.ipynb`, `07_dl_data_checks.ipynb`, `08_latent_gmm.ipynb`, `09_hybrids_ablation.ipynb`, `06b_final_comparison_phase2.ipynb` |
| **LaTeX Report Modules** | 10 modular section files in `reports/ieee_report/sections/` | ✅ PASS | `p2_introduction`, `p2_dl_literature`, `p2_dl_data_pipeline`, `p2_architecture`, `p2_ae_results`, `p2_hybrid_theory`, `p2_latent_gmm_results`, `p2_hybrid`, `p2_ablation`, `p2_validation` |
| **Bibliography Integrity** | All citations resolve; `[VERIFY]` markers removed | ✅ PASS | 58 unique verified entries in `reports/ieee_report/refs_phase2.bib`; Bourlard & Kamp (1988) added; 0 missing citations |

---

## 3. Git Commit Cadence & Author Audit

In accordance with project guidelines, we audit the genuine Git commit history. **No commit timestamps were fabricated or backdated.**

| Team Member | Git Author Names | Total Phase 2 Commits | Calendar Days | Commit Details & Genuine Audit Status |
|---|---|---|---|---|
| **Krishna Pramod Dubale** | `Krishna Pramod Dubale` | 3 | 1 day (2026-11-10) | Commits: `050459a`, `f4bbb4a`, `564a954`. *Audit Flag:* Work was committed in 3 substantial sittings on 2026-11-10. Documented honestly without artificial date tampering. |
| **Vani Rudra** | `vanosaur`, `Vani Rudra` | 3 | 1 day (2026-11-11) | Commits: `b7ab893`, `a904651`, `09816d7`. *Audit Flag:* Completed autoencoder architecture, 5-seed training, and LaTeX report on 2026-11-11. |
| **Apoorva Choudhary** | `Apoorv Choudhary`, `Apoorva choudhary` | 3 | 2 days (2026-11-11, 2026-11-12) | Commits: `f5c16c5` (Nov 11), `51d5f86` (Nov 12), `9c3fbe6` (Nov 12). |
| **Prachee Dhar** | `pracheedhar`, `Prachee Dhar` | 3 | 1 day (2026-11-15) | Commits: `a2466ac`, `19a117c`, `931e7ab`. *Audit Flag:* Completed hybrid fusion, DAGMM-lite, and master ablation on 2026-11-15. |
| **Kushal Sarkar** | `Kushal Sarkar`, `Kushal425` | 3 | Genuine calendar timestamps | Scaffold, final master integration, report generation, and presentation assembly. |

> ⚠️ **Cadence Disclosure (Non-Blocking):** Team members completed their multi-sitting commits on intensive development sprints rather than three separate calendar days. Rather than falsifying history via `GIT_COMMITTER_DATE` (which violates academic integrity guidelines), this distribution is reported transparently.

---

## 4. Master 10-Model Empirical Comparison Table

All numbers below were computed directly from `results/scores/*.npy` by `src/final_phase2.py` on the held-out test split ($N=56,962$, 75 frauds, base rate $0.1317\%$):

| Model Key | Display Name | Phase | Val PR-AUC | Test PR-AUC | 95% Bootstrap CI | Test ROC-AUC | Recall @ 0.1% FPR | Recall @ 0.5% FPR | Precision @ K=100 |
|---|---|---|---|---|---|---|---|---|---|
| `pca` | **PCA Reconstruction** | P1 | **0.7066** | **0.7425** | **[0.6415, 0.8368]** | 0.9303 | **76.00%** | 77.33% | **57.0%** |
| `ae_dae` | **Denoising Autoencoder (DAE)** | P2 | 0.2935 | **0.4818** | [0.3707, 0.6166] | 0.9322 | 70.67% | 77.33% | 53.0% |
| `hybrid_fusion` | **H2: Score Fusion** | P2 | 0.2935 | **0.4818** | [0.3707, 0.6166] | 0.9322 | 70.67% | 77.33% | 53.0% |
| `hybrid_dagmmlite`| **H3: DAGMM-lite** | P2 | 0.3913 | **0.4541** | [0.3524, 0.5884] | 0.9127 | 68.00% | **80.00%** | 48.0% |
| `lof` | **Local Outlier Factor** | P1 | 0.4069 | 0.4478 | [0.3447, 0.5822] | 0.9048 | 65.33% | 72.00% | 47.0% |
| `ae` | **Standard Deep AE** | P2 | 0.2272 | 0.3574 | [0.2618, 0.4848] | 0.9021 | 53.33% | 68.00% | 40.0% |
| `gmm` | **Full-Space GMM (K=4)** | P1 | 0.1038 | 0.2679 | [0.1843, 0.3800] | 0.9391 | 33.33% | 66.67% | 31.0% |
| `hybrid_latentgmm`| **H1: Latent-Space GMM** | P2 | 0.1159 | 0.1724 | [0.1329, 0.2310] | 0.9398 | 24.00% | 74.67% | 30.0% |
| `ocsvm` | **One-Class SVM** | P1 | 0.0613 | 0.1322 | [0.1030, 0.1784] | **0.9604** | 0.00% | 69.33% | 16.0% |
| `iforest` | **Isolation Forest** | P1 | 0.0330 | 0.0577 | [0.0444, 0.0787] | 0.9567 | 1.33% | 34.67% | 7.0% |

---

## 5. Answers to Central Research Questions

### 1. Does Deep Learning Outperform Classical GMM?
**Verdict: YES.**  
The Denoising Autoencoder ($\text{PR-AUC} = 0.4818$) significantly outperforms full-space GMM ($\text{PR-AUC} = 0.2679$).  
- **Paired Bootstrap Difference:** $\Delta = +0.2139$ [95% CI: $+0.1177, +0.3124$], which excludes zero ($p < 0.05$).  
- At a tight false-alarm constraint ($\text{FPR} \le 0.1\%$), DAE detects **70.67%** of frauds, more than doubling GMM's catch rate (**33.33%**).

### 2. Does Deep Learning Outperform Linear PCA?
**Verdict: NO.**  
Linear PCA reconstruction error remains the strongest standalone detector ($\text{PR-AUC} = 0.7425$).  
- **Paired Bootstrap Difference:** $\text{DAE} - \text{PCA} = -0.2607$ [95% CI: $-0.3721, -0.1492$], $p < 0.05$.  
- **Scientific Explanation:** The ULB benchmark features $V_1$--$V_{28}$ are already derived from PCA. Legitimate transaction variations are predominantly linear. Deep autoencoders possess universal function approximation capacity, which inadvertently allows them to partially reconstruct subtle fraud patterns (over-generalization; Gong et al., 2019). Linear PCA has zero capacity to bend around anomalies, creating steeper residual penalties for fraudulent deviations.

### 3. Does Latent-Space GMM (H1) Resolve the Curse of Dimensionality?
**Verdict: NO.**  
While bottleneck compression ($k=8$) reduces covariance parameter complexity by **92.3%** (from 465 to 36 parameters per component), H1 achieves a test PR-AUC of only **0.1724**.  
- **The Theoretical Root Cause:** The non-linear encoder is not volume-preserving. Under non-linear dimensionality reduction, latent density does not equal data density without the log-determinant Jacobian penalty ($\log p(\mathbf{x}) = \log p(\mathbf{z}) - \frac{1}{2}\log\det(J^T J)$). Outlier transactions in data space are mapped into interior modes of the latent space, producing low anomaly scores.  
- Furthermore, fraud transactions in credit streams are predominantly *off-manifold* (large reconstruction residuals) rather than on-manifold low-density events.

### 4. Does Hybrid Modeling Provide Value?
**Verdict: YES, in Operational Catch Rate.**  
While H2 Score Fusion selects $w=0.0$ on validation (converging to pure DAE), **H3 (DAGMM-lite)** achieves the **highest recall on test frauds at FPR $\le 0.5\%$ of any model: 80.00%** (catching 60 of 75 frauds vs PCA's 58 frauds). Concatenating latent coordinates with reconstruction error and cosine angle distortion enables the mixture model to capture both off-manifold distance and directional shifts simultaneously.

---

## 6. Flagged Items & Known Limitations

1. **Finite Sample Size of Test Frauds ($N=75$):**  
   While the test split contains 56,962 transactions, the extreme class imbalance ($0.13\%$) means only 75 fraud events exist. Consequently, 95% bootstrap confidence intervals span approximately $\pm 0.10$ to $\pm 0.12$ PR-AUC points.
2. **Feature Anonymization:**  
   Because features $V_1$--$V_{28}$ are pre-transformed via PCA by the dataset providers, domain-specific feature engineering (e.g., merchant risk categories, geographic velocity) cannot be constructed.
3. **Short Temporal Window (48 Hours):**  
   The dataset captures transactions over 2 days. Long-term concept drift, weekly seasonality, and holiday shopping surges cannot be modeled.
4. **Academic Research Prototype Disclaimer:**  
   This implementation is an academic research prototype intended to benchmark representation learning under class imbalance; it does not replace a production bank fraud engine.

---

## 7. Verification & Reproducibility Commands

To reproduce the complete Phase 2 verification independently on a clean environment:

```bash
# 1. Run all unit, invariant, and score contract tests (139 tests)
pytest tests/ -v

# 2. Run the Phase 2 master evaluation and figure generator
python3 src/final_phase2.py

# 3. Verify all score files and invariants
python3 -c "import numpy as np; [print(f, np.load(f'results/scores/{f}').shape) for f in sorted(os.listdir('results/scores')) if f.endswith('.npy')]"
```
