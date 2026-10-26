# Master Viva Guide — Team T033, Project 26
**Card Fraud Triage: Gaussian Mixture Density Scoring vs Classical Baselines**  
*Phase 1: Classical Machine Learning & Statistical Anomaly Detection*  
*Team Members:* Kushal Sarkar (Lead), Prachee Dhar, Apoorva Choudhary, Vani Rudra, Krishna Pramod Dubale  
*Faculty Advisors:* Shobhit Johri, Arun Chauhan, Ashwin Tewary (Rishihood University)

---

## 1. Executive Summary & 2-Minute Project Pitch (All Members)

> **The 30-Second Elevator Pitch:**  
> "We treat credit card fraud detection as an unsupervised anomaly detection problem because fraudulent transactions are ultra-rare (only $0.1727\%$ in the ULB benchmark). Supervised classifiers collapse toward the majority class under extreme imbalance and fail when ground-truth labels are delayed by days or weeks. Instead, our models train exclusively on verified legitimate transactions to learn the normal spending manifold. When scoring test transactions, deviation from normality indicates potential fraud. Our primary model is a Gaussian Mixture Model (GMM) scored via Negative Log-Likelihood, benchmarked against PCA Reconstruction Error, Isolation Forest, One-Class SVM, and Local Outlier Factor. Because standard accuracy and ROC-AUC are deceptively inflated by billions of true negatives, we evaluate models via Precision-Recall AUC (PR-AUC), recall at operational false-positive rate budgets ($0.1\%, 0.5\%, 1.0\%$), and precision at review capacity."

### Core Project Numbers to Know by Heart
* **Total Transactions:** $284,807$ over 48 continuous hours (September 2013, European cardholders).
* **Total Frauds:** $492$ transactions ($0.1727\%$ base rate, or 1 fraud per 578 transactions).
* **Chronological 60/20/20 Splits:**
  * **Train:** $170,884$ rows ($170,524$ legit, $360$ fraud, base rate $0.2107\%$) — $[0, 120\,396\,\text{s}]$
  * **Validation:** $56,961$ rows ($56,904$ legit, $57$ fraud, base rate $0.1001\%$) — $[120\,396, 145\,247\,\text{s}]$
  * **Test:** $56,962$ rows ($56,887$ legit, $75$ fraud, base rate $0.1317\%$) — $[145\,248, 172\,792\,\text{s}]$
* **Model Benchmark Winner:** **PCA Reconstruction Error** achieves top test PR-AUC of **$0.7425$** ($77.33\%$ recall at $\text{FPR}=0.5\%$).
* **GMM Density Baseline:** Achieves test PR-AUC of **$0.4142$** ($69.33\%$ recall at $\text{FPR}=0.5\%$), outperforming Random ($0.0015$), Amount-only ($0.0034$), and Isolation Forest ($0.0577$).

---

## 2. Universal Viva Questions & Theoretical Traps

### Q1: Why frame credit card fraud as anomaly detection instead of supervised classification?
1. **Extreme Class Imbalance:** Fraud represents only $0.1727\%$ of transactions. Standard classification losses (cross-entropy) collapse to majority voting: predicting 100% legitimate yields $99.83\%$ accuracy while missing every single fraud event.
2. **Verification Lag & Chargeback Asymmetry:** Ground truth is not available in real-time. Legitimate transactions clear silently; fraudulent transactions are discovered days or weeks later following cardholder complaints or chargebacks. Supervised models trained on recent unverified data suffer severe label noise.
3. **Concept Drift & Adversarial Shift:** Fraud syndicates continually alter tactics (spending velocity, micro-charges, merchant categories). Anomaly detection models the stable manifold of legitimate customer habits, flagging novel deviations without requiring prior historical examples of a specific attack pattern.

### Q2: Why is accuracy useless, and why does ROC-AUC fail under severe class imbalance?
* **Accuracy Fallacy:** $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$. When $TN = 56,887$ and $P = 75$, a dummy model predicting 0 for all rows achieves $99.87\%$ accuracy but has $0\%$ recall.
* **ROC-AUC Deception:** ROC curves plot True Positive Rate against False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). Because $TN$ is massive ($56,887$), flagging $285$ false alarms yields an FPR of only $285 / 57,000 \approx 0.005$ ($0.5\%$). The ROC curve barely flinches and stays above $0.95$ (e.g. Isolation Forest has ROC-AUC $0.9567$), even though its PR-AUC is terrible ($0.0577$) and investigators would be overwhelmed with false alarms.
* **PR-AUC (Average Precision):** Plots Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) against Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). A random model scores exactly the base rate ($0.0013$ on test). PR-AUC directly penalizes false alarms.

---

## 3. Kushal Sarkar — Scaffold, Leakage Prevention & System Architecture

### Responsibilities
* Repository scaffolding, configuration management, data pipelines, chronological splitting, metrics calculation contracts, test suite, and final integration.

### Key Technical Talking Points
1. **Strict Zero-Leakage Architecture:**
   * **Rule 1 (Chronological Partitioning):** Transactions are ordered by time ($t \in [0, 172\,792]$). Splitting is sequential (60/20/20) without shuffling. Future transactions never leak into the past.
   * **Rule 2 (Legitimate-Only Fitting):** All preprocessors (`StandardScaler`) and anomaly models are fitted **strictly on training legitimate transactions** (`Class == 0`). Fraud labels are never seen during training.
   * **Rule 3 (Validation-Frozen Thresholds):** All operational cutoffs (FPR budgets, top-$K$ capacities) are computed on the validation split and frozen before applying to the test set.
2. **Score Contract:**
   * Every model exports a 1-D `float32` array to `results/scores/{model}_{split}.npy`.
   * **Contract Rule:** Higher score = more anomalous = higher fraud likelihood. All models conform to this contract (e.g., GMM exports $-\ln p(x)$, PCA exports reconstruction residual $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$).
3. **Automated Testing & Code Quality:**
   * Authored `test_no_leakage.py`, `test_metrics.py`, and `test_score_contract.py`. Full test suite achieves **80/80 passing tests** in automated CI (`pytest`).

---

## 4. Prachee Dhar — Literature Review, EDA & Dataset Quality

### Responsibilities
* Literature synthesis ($\ge 12$ verified papers), exploratory data analysis, class imbalance analysis, temporal drift KS-testing, and Gaussianity audits.

### Key Technical Talking Points
1. **Dataset Integrity & Duplicate Analysis:**
   * $284,807$ rows $\times$ $31$ numeric columns, $0$ missing values.
   * Discovered **$1,081$ exact duplicate records** ($0.38\%$ of dataset), accounting for $1,854$ duplicate rows ($1,822$ legitimate, $32$ fraudulent). Exactly one duplicate group straddled class labels (identical features with different labels). Duplicates are retained to reflect real raw telemetry.
2. **Amount & Time Dynamics:**
   * **Amount:** Legitimate mean is $\$88.29$ (median $\$22.00$); fraud mean is higher at $\$122.21$ but median is much lower ($\$9.25$). Frauds cluster either in micro-charges (card testing) or large cash-outs. Heavy-tailed distribution ($11.2\%$ outliers beyond $Q3 + 1.5\text{IQR}$) necessitates $\ln(1 + \text{Amount})$ transformation.
   * **Time:** Circadian patterns show transactions drop between 2 AM and 6 AM, but fraud maintains steady nocturnal velocity, exploiting off-peak periods when manual authorization desks are understaffed.
3. **Gaussianity Audit (Testing GMM Assumptions):**
   * Shapiro-Wilk tests rejected normality for all 28 PCA features ($p < 0.05$).
   * Features exhibit extreme leptokurtosis (V2 kurtosis $114.97$, V8 kurtosis $261.68$). Only 9 of 28 features satisfy mild normality criteria ($|\text{skew}| < 1, |\text{kurt}| < 3$).
   * **Why this matters for GMM:** Single-component Gaussians ($K=1$) fail completely. Multi-component GMMs ($K \ge 3$) are mandatory to cover heavy-tailed clusters.

---

## 5. Apoorva Choudhary — Feature Engineering & GMM Density Modeling

### Responsibilities
* Feature engineering pipelines, GMM formulation, Expectation-Maximization (EM) derivation, AIC/BIC hyperparameter search, and instability studies.

### Key Technical Talking Points
1. **Why PCA Restricts Feature Engineering:**
   * Features V1–V28 are PCA projections created by ULB to preserve European customer privacy.
   * Because $V_i$ are uninterpretable linear combinations of proprietary banking features, speculative feature crosses ($V_i \times V_j$) lack domain meaning. Furthermore, principal components are orthogonal ($r \approx 0$).
   * **Our Solution:** Mathematical and geometric engineering:
     1. $\ln(1 + \text{Amount})$ to compress heavy tails.
     2. Cyclic time encoding: $\sin(2\pi t / 86\,400)$ and $\cos(2\pi t / 86\,400)$.
     3. **Joint Geometric Norms:** $\|\mathbf{V}\|_2 = \sqrt{\sum V_i^2}$, 3-sigma outlier count $\sum \mathbb{I}(|V_i| > 3)$, and $\max |V_i|$.
     4. Pruning uninformative features ($V_{22}, V_{23}, V_{25}$ had KS separation $<0.05$). Pruning boosted validation PR-AUC to $0.06934$ ($+48.8\%$ over base).
2. **EM Algorithm Derivation:**
   * Maximizing incomplete log-likelihood $\ln p(\mathbf{X} \mid \boldsymbol{\theta})$ directly is non-convex due to the sum inside the log.
   * **E-Step:** Compute responsibilities $\gamma_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$.
   * **M-Step:** Analytically update mixing weights $\pi_k = \frac{N_k}{N}$, means $\boldsymbol{\mu}_k = \frac{1}{N_k}\sum \gamma_{nk}\mathbf{x}_n$, and covariances $\boldsymbol{\Sigma}_k$.
   * **Monotonic Convergence:** Proven via Jensen's inequality and the Evidence Lower Bound (ELBO): $\ln p(\mathbf{X} \mid \boldsymbol{\theta}^{(t+1)}) \ge \ln p(\mathbf{X} \mid \boldsymbol{\theta}^{(t)})$.
3. **AIC vs BIC Model Selection:**
   * Evaluated 120 configurations. AIC penalizes parameters by $2p$, which under-penalizes in large samples ($N=170\,524$) and over-clusters.
   * BIC penalizes by $p \ln(N)$, finding global minimum at $K=16$, full covariance.
   * **Divergence:** While BIC selected $K=16$ ($\text{reg}=10^{-6}$), validation triage favored $K=12$ with stronger regularization ($\text{reg}=10^{-2}$), which prevents the model from assigning spuriously low densities to legitimate boundary transactions.
4. **Covariance Singularity & Regularization:**
   * If a component collapses onto a single data point ($\boldsymbol{\mu}_k \to \mathbf{x}_m$), variance shrinks to zero, $|\boldsymbol{\Sigma}_k| \to 0$, and likelihood explodes to $+\infty$. Regularization adds $\epsilon_{\text{cov}} \mathbf{I}$ (`reg_covar = 1e-2`) to enforce minimum eigenvalues and guarantee positive definiteness.

---

## 6. Vani Rudra — Classical Anomaly Detection Baselines

### Responsibilities
* Implementation, hyperparameter sweeps, and benchmarking of classical baselines: PCA Reconstruction Error, Isolation Forest, One-Class SVM, Local Outlier Factor (LOF), and naive reference baselines.

### Key Technical Talking Points
1. **PCA Reconstruction Error (The Overall Winner):**
   * Projects transaction $\mathbf{x}$ onto top-$k$ principal components: $\hat{\mathbf{x}} = \mathbf{U}_k \mathbf{U}_k^\top \mathbf{x}$.
   * Anomaly score: $s(\mathbf{x}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$.
   * **Why it wins (PR-AUC 0.7425):** The 28 V-features already represent PCA dimensions of legitimate activity. Legitimate transactions reconstruct with near-zero error from a low-rank subspace, whereas frauds deviate sharply into minor residual components.
2. **Isolation Forest (The High-Dimensional Trap):**
   * Recursively partitions space via random axis-aligned splits. Anomalies require fewer splits to isolate (shorter tree path length $h(\mathbf{x})$).
   * **Why it failed (PR-AUC 0.0577):** In 31 dimensions where fraud patterns lie along correlated diagonal manifolds rather than extreme coordinate outliers, random axis-aligned partitioning suffers from the curse of dimensionality and fails to isolate anomalies early.
3. **One-Class SVM (RBF Kernel):**
   * Finds a maximum-margin hyperplane in RKHS enclosing normal points while maximizing distance from the origin. Score: signed distance $-f(\mathbf{x})$.
   * **Computational Barrier:** $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$ complexity. Fitting on full 170k rows took $>58$ seconds even on a 40k subsample. PR-AUC is $0.1322$; decision boundary at strict budgets ($\text{FPR}=0.1\%$) flagged zero frauds.
4. **Local Outlier Factor (LOF):**
   * Compares local reachability density of a point against its $k$-nearest neighbors. Fit in `novelty=True` mode on legitimate training rows.
   * Achieved solid test PR-AUC of **$0.4478$** ($72.0\%$ recall at $\text{FPR}=0.5\%$), ranking second behind PCA.

---

## 7. Krishna Pramod Dubale — Evaluation Protocol & Threshold Triage

### Responsibilities
* Evaluation framework, metric contracts, stratified bootstrap confidence intervals, operational threshold policies (FPR budgets, review capacity), and cost curve analysis.

### Key Technical Talking Points
1. **Stratified Bootstrap Confidence Intervals:**
   * Why bootstrap? The test set contains only $75$ frauds. A single misclassified fraud changes recall by $1.33\%$. Metrics without uncertainty intervals are misleading.
   * **Procedure ($B=2000$ resamples):** Separate test set into fraud pool ($75$) and legitimate pool ($56,887$). Resample each pool with replacement, compute metric, and extract 2.5th and 97.5th percentiles.
   * Result: PCA test PR-AUC is $0.7425$ with $95\%$ CI $[0.6415, 0.8368]$. Difference between PCA and all other models is statistically significant ($p < 0.05$).
2. **Operational Threshold Policies:**
   * Rather than arbitrary 0.5 classification cutoffs, banks use two real-world operating policies:
     1. **FPR Budget Policy:** "We tolerate flagging at most $0.5\%$ of legitimate volume." Derived as the 99.5th percentile of legitimate validation scores, frozen, and evaluated on test. (PCA catches $77.33\%$ of fraud at $0.5\%$ FPR).
     2. **Capacity Queue Policy:** "Our investigation desk can review at most $K=100$ alerts per shift." Precision@100 measures analyst time efficiency. (GMM achieves $37.0\%$ precision in top-100; PCA achieves $>60\%$).
3. **Cost-Sensitive Decision Theory:**
   * Cost function: $\text{Total Cost} = C_{\text{FP}} \cdot \text{FP} + C_{\text{FN}} \cdot \text{FN}$.
   * We sweep cost ratio $r = C_{\text{FN}} / C_{\text{FP}} \in [10, 500]$. As missed fraud cost increases, the optimal threshold decreases, aggressively flagging more transactions to minimize high-loss false negatives.

---

## 8. Master Performance Summary Table (All Models)

| Model | Anomaly Paradigm | PR-AUC (Test) | 95% Bootstrap CI | ROC-AUC (Test) | Recall @ FPR=0.5% | Prec @ 0.5% Cap | Train Time (s) |
|---|---|---|---|---|---|---|---|
| **PCA Recon.** | Linear Residual Subspace | **0.7425** | $[0.6415, 0.8368]$ | $0.9303$ | **77.33%** | **20.00%** | **0.8s** |
| **LOF** | Local Density Ratio | $0.4478$ | $[0.3447, 0.5822]$ | $0.9048$ | $72.00\%$ | $18.60\%$ | $24.2\text{s}$ |
| **GMM** | Parametric Density ($-\ln p$) | $0.4142$ | $[0.3050, 0.5403]$ | $0.9274$ | $69.33\%$ | $18.25\%$ | $22.1\text{s}$ |
| **OC-SVM** | Kernel Hyperplane Distance | $0.1322$ | $[0.1030, 0.1784]$ | $0.9604$ | $69.33\%$ | $16.49\%$ | $58.4\text{s}$ |
| **Iso. Forest** | Tree Partitioning Depth | $0.0577$ | $[0.0444, 0.0787]$ | $0.9567$ | $34.67\%$ | $7.72\%$ | $14.1\text{s}$ |
| *Amount (Ref)* | Transaction Magnitude | $0.0034$ | $[0.0026, 0.0051]$ | $0.7095$ | $1.33\%$ | $0.35\%$ | $<0.1\text{s}$ |
| *Random (Ref)* | Uniform Pseudo-Random | $0.0015$ | $[0.0011, 0.0037]$ | $0.4840$ | $1.33\%$ | $0.35\%$ | $<0.1\text{s}$ |

---

## 9. Live Demo & Code Walkthrough Checklist

During the viva, if asked to demonstrate code execution:
1. **Show automated tests:** Run `pytest` $\implies$ proves strict non-leakage, metric contracts, and score invariants ($80/80$ passing).
2. **Show final summary generation:** Run `python3 src/final.py` $\implies$ recreates all comparison tables and PR curves.
3. **Show live model inference:**
   ```python
   import joblib, numpy as np
   # Load serialized model
   model = joblib.load("results/models/gmm.joblib")
   sample = np.random.randn(1, 31)
   score = -model.score_samples(sample)[0]
   print(f"Transaction Anomaly Score (NLL): {score:.4f}")
   ```
4. **Show error analysis:** Open `results/tables/gmm_failure_analysis.csv` to explain false negatives (frauds whose amounts and PCA values mimicked legitimate transactions).
