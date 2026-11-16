# Viva Notes — Phase 2 Final Integration & Research Defense
**Author:** Kushal Sarkar (Team Lead & Integration Owner)  
**Team:** T033 | Project 26: Card Fraud Triage  
**Date:** 2026-11-16  

---

## 1. Executive Summary & 2-Minute Phase 2 Pitch

> "In Phase 1, we established that linear PCA reconstruction error achieved the strongest anomaly detection performance ($\text{PR-AUC} = 0.7425$), outperforming full-space GMM ($\text{PR-AUC} = 0.4142$) and classical baselines.  
> In Phase 2, we addressed two fundamental hypotheses:  
> 1. Can deep non-linear autoencoders learn curved transaction manifolds better than linear PCA?  
> 2. Can modeling the latent bottleneck distribution via Gaussian Mixture Models (H1, H2, H3) eliminate the curse of dimensionality while preserving explicit density estimation?  
>
> Our empirical findings on the untouched chronological test set ($N=56,962$, 75 frauds) revealed:  
> - **Denoising Autoencoder (DAE, $\sigma=0.2$) significantly outperforms full-space GMM** ($\text{PR-AUC} = 0.4818$ vs $0.2679 / 0.4142$, paired difference $\Delta = +0.2139$, $p < 0.05$) and doubles fraud catch rate at tight operational budgets ($70.67\%$ recall vs $33.33\%$ at $\text{FPR} \le 0.1\%$).  
> - **Linear PCA remains the strongest overall model ($\text{PR-AUC} = 0.7425$).** Because the ULB transaction features $V_1$--$V_{28}$ are already derived from PCA, legitimate data occupies a linear subspace. Deep autoencoders possess universal function approximation capacity, which inadvertently leads to partial over-generalization to subtle fraud patterns (Gong et al., 2019).  
> - **Latent-space GMM (H1) underperforms ($\text{PR-AUC} = 0.1724$).** The non-linear encoder is not volume-preserving; without the Jacobian determinant correction, latent density does not equal data density. Furthermore, card frauds are predominantly *off-manifold* (large reconstruction residuals) rather than *on-manifold low-density* events.  
> - **DAGMM-lite (H3) achieves the highest operational recall at FPR $\le 0.5\%$:** catching **80.00%** of test frauds (60 of 75 frauds vs PCA's 58), demonstrating that joint modeling of latent coordinates and reconstruction distance captures subtle edge cases."

---

## 2. Answers to Essential Phase 2 Viva Questions (Q17 – Q39)

### Q17: Why did we test an Autoencoder for credit card fraud detection?
- **Theoretical Motivation:** PCA is limited to linear orthogonal projections. An undercomplete deep autoencoder ($f_{\boldsymbol{\theta}}: \mathbb{R}^d \to \mathbb{R}^k$, $g_{\boldsymbol{\phi}}: \mathbb{R}^k \to \mathbb{R}^d$) uses non-linear activations (LeakyReLU) to learn continuous curved sub-manifolds embedded within ambient transaction space.
- **Anomaly Detection Protocol:** Trained strictly on nominal legitimate transactions, the network allocates its representational capacity toward normal correlation structures. When presented with fraudulent transactions, it cannot reconstruct them accurately, generating large reconstruction residuals $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$.

---

### Q18: What is the mathematical connection between Linear Autoencoders and PCA?
- Minimizing squared reconstruction error $\mathbb{E}[\|\mathbf{x} - \mathbf{W}_2 \mathbf{W}_1 \mathbf{x}\|_2^2]$ with a linear bottleneck of rank $k$ spans the exact same subspace as projecting onto the top-$k$ principal eigenvectors of the data covariance matrix (Bourlard & Kamp 1988, Baldi & Hornik 1989).
- **Empirical Proof:** In `src/autoencoder.py`, we implemented a linear AE sanity check. At bottleneck $k=8$, linear AE reconstruction scores correlate at Pearson $r = 0.9933$ with PCA reconstruction error.

---

### Q19: Why does Denoising Autoencoder (DAE) outperform standard Undercomplete AE?
- **The Score Matching Connection (Vincent 2011):** Training a network to reconstruct uncorrupted inputs from stochastically corrupted vectors $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$ ($\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$) forces the vector $(\hat{\mathbf{x}} - \tilde{\mathbf{x}})$ to point along the score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$.
- **Resistance to Over-Generalization:** Standard AEs can memorize high-frequency transaction quirks. DAE forces the network to learn the orthogonal projection vector back to the data manifold, expanding the basin of attraction for legitimate data and sharpening the reconstruction error on frauds.
- **Empirical Impact:** DAE ($\sigma=0.2$) improved test PR-AUC from **0.3574** (AE) to **0.4818** ($+34.8\%$ relative gain) and Recall@0.1% FPR from **53.33%** to **70.67%**.

---

### Q20: Why did linear PCA outperform Deep Autoencoders on this dataset?
1. **PCA Pre-Anonymized Dataset:** The original ULB dataset features $V_1$--$V_{28}$ are already the principal components of an internal bank dataset. Legitimate transaction variations are predominantly linear.
2. **Inductive Bias & Over-Generalization:** Deep neural networks have excess representational capacity. As Gong et al. (ICCV 2019) demonstrated, without memory constraints, deep decoders partially reconstruct anomalies. Linear PCA has zero capacity to bend around anomalies, creating steeper residual penalties for off-subspace deviations.
3. **Statistical Parsimony:** PCA has zero trainable non-linear parameters; it requires no stochastic gradient descent, has no initialization seed variance, and does not suffer from local minima.

---

### Q21: What is the "Change-of-Variables Caveat" in Latent GMM (H1)?
- When mapping $\mathbf{x} \in \mathbb{R}^d$ to $\mathbf{z} \in \mathbb{R}^k$ via non-linear encoder $f_{\boldsymbol{\theta}}$, the probability density transform requires a square, invertible transformation with Jacobian $J$:
  $$p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f(\mathbf{x})) \cdot |\det J_f(\mathbf{x})|$$
- In an undercomplete autoencoder, $k \ll d$ (e.g., $8 \ll 30$), so the mapping is non-invertible and non-volume-preserving. Evaluating $-\log p(\mathbf{z})$ evaluates density in the *latent representation space*, NOT data space $\mathbb{R}^d$.
- An outlier transaction in $\mathbb{R}^{30}$ can be mapped by the encoder to the center of the latent distribution $\mathbf{z} \approx \mathbf{0}$, causing the GMM to assign it high density (low anomaly score).

---

### Q22: Why did Latent GMM (H1) fail empirically?
1. **Latent Non-Gaussianity:** Henze-Zirkler and Mardia tests reject multivariate normality on latent codes $\mathbf{z} \in \mathbb{R}^8$ ($p < 10^{-6}$). The latent distributions have heavy tails and multimodal skewness.
2. **Off-Manifold vs On-Manifold Nature of Card Fraud:** Card frauds deviate primarily in orthogonal feature directions (high reconstruction error $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$). Once projected into the bottleneck manifold, their coordinates overlap with normal transactions.
3. **Result:** H1 test PR-AUC was only **0.1724**, lower than both standalone AE (**0.3574**) and classical GMM (**0.4142**).

---

### Q23: How does Hybrid H3 (DAGMM-lite) overcome H1's limitation?
- Inspired by Zong et al. (ICLR 2018), H3 decouples representation learning from density estimation to prevent EM covariance collapse.
- It concatenates latent codes $\mathbf{z} \in \mathbb{R}^8$ with explicit reconstruction metrics:
  $$\mathbf{v} = [\mathbf{z}, \text{MSE}(\mathbf{x}, \hat{\mathbf{x}}), \text{CosineDistance}(\mathbf{x}, \hat{\mathbf{x}})] \in \mathbb{R}^{10}$$
- A GMM is fitted on $\mathbf{v}$ on training legitimate data. Anomaly score: $s(\mathbf{x}) = -\log p(\mathbf{v})$.
- **Result:** Captures both off-manifold distance ($\text{MSE}$) and angle distortion ($\text{CosineDist}$). It achieves the **highest recall at 0.5% FPR across all models: 80.00%** (catching 60 of 75 frauds).

---

### Q24: How did we ensure strict zero test leakage throughout Phase 2?
1. **Chronological Splitting:** 60% train ($[0, 120\,396\,\text{s}]$), 20% val ($[120\,396, 145\,247\,\text{s}]$), 20% test ($[145\,248, 172\,792\,\text{s}]$). No random shuffling.
2. **Legitimate-Only Training:** All neural networks (AE, DAE) and latent GMMs were trained strictly on legitimate records (`Class == 0`, $N=170,524$).
3. **Validation Early Stopping:** Early stopping monitored reconstruction loss on *validation legitimate* records; no fraud labels were used for stopping.
4. **Validation-Frozen Fusion Weights:** In H2 score fusion, grid weights $w \in [0, 1]$ were chosen strictly by validation PR-AUC ($w=0.0$ selected).
5. **Validation-Frozen Thresholds:** Operational alert cutoffs (FPR budgets 0.1%, 0.5%, 1.0%) were determined on validation and frozen before scoring test data.

---

### Q25: How does this system translate to operational banking deployment?
- **Analyst Workload:** A fraud investigation team has bounded review bandwidth (e.g., 100 cases per shift).
- **At Top 100 Alerts ($K=100$):**
  - Linear PCA flags **57 true frauds** (Precision = $57.0\%$).
  - DAE flags **53 true frauds** (Precision = $53.0\%$).
  - DAGMM-lite flags **48 true frauds** (Precision = $48.0\%$).
  - Classical GMM flags **31 true frauds** (Precision = $31.0\%$).
- **Inference Speed:** DAE CPU latency is **$12.4\text{ ms}$ per 10,000 transactions** ($1.24\,\mu\text{s}$ per transaction), which is well within the typical credit card authorization SLA of $100\text{ ms}$.
- **Proposed Architecture:** A two-stage ensemble:
  - Stage 1: Linear PCA fast filter eliminates $99\%$ of clear legitimate transactions.
  - Stage 2: DAE & DAGMM-lite inspect the remaining $1\%$ to detect non-linear evasion attempts.

---

## 3. Defense Checklist for Kushal Sarkar (Team Lead)
- [x] Verified git branch: `feat/lead-final-p2`.
- [x] Confirmed all four teammate PRs (#13 Krishna, #14 Vani, #15 Apoorva, #16 Prachee) are merged on `main`.
- [x] Verified 139 passing automated unit, invariant, and contract tests.
- [x] Confirmed zero fabricated numbers across all CSV tables, figures, and LaTeX macros.
- [x] Verified that Phase 1 reports and code remain 100% intact and reproducible.
