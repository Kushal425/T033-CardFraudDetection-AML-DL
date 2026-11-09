# Phase 2 Presentation Slides Content: Deep Learning Literature & Data Pipeline

**Author:** Krishna Pramod Dubale (Enrollment: 2401010239)  
**Team:** T033, Project 26 — Card Fraud Triage  
**Deliverable:** Phase 2 Data & Literature Section (`docs/slides_phase2_data_section.md`)  

---

## Slide 1: Deep Learning Literature Review & The Manifold Gap

### Context & Motivation
- **Phase 1 Takeaway:** PCA Reconstruction Error dominated Phase 1 ($\text{PR-AUC} = 0.7425$), outperforming full-space GMM ($\text{PR-AUC} = 0.4142$).
- **Theoretical Limitation of PCA:** Linear projection is restricted to flat linear hyperplanes; real transaction patterns form non-linear, multi-scale manifolds.
- **Deep Autoencoders (AE):** Non-linear neural networks discover curved coordinate transforms (Hinton & Salakhutdinov, 2006 `[VERIFY]`).

### Key Theoretical Foundations
- **Score Matching in Denoising AEs:** Training DAEs with Gaussian noise is asymptotically equivalent to learning the score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ of the normal data distribution (Vincent, 2011 `[VERIFY]`).
- **Over-Generalization Pathology:** High-capacity undercomplete autoencoders can inadvertently reconstruct out-of-distribution anomalies (Gong et al., 2019 `[VERIFY]`), requiring tight bottlenecks ($k=8$) and strict regularization.
- **The Decoupled Hybrid Rationale:** Reconstruction error measures *orthogonal distance* to the manifold ($\|\mathbf{x} - \hat{\mathbf{x}}\|^2$), while latent GMM estimates *probability density* *on* the manifold ($-\log p(\mathbf{z})$) (Zong et al., 2018 `[VERIFY]`).

---

## Slide 2: Deep Learning Data Pipeline & PyTorch Architecture

### Chronological Partitioning Protocol
- **48-Hour Continuous Stream:** 284,807 transactions chronologically partitioned into non-overlapping windows:
  - **Train (60%):** 170,884 transactions ($t \le 120,396\,\text{s}$)
  - **Validation (20%):** 56,961 transactions ($120,396\,\text{s} < t \le 145,247\,\text{s}$)
  - **Test (20%):** 56,962 transactions ($t > 145,248\,\text{s}$)
- **Why Chronological?** Random $k$-fold cross-validation causes massive temporal look-ahead leakage, corrupting real-world fraud triage evaluations (Lucas et al., 2020 `[VERIFY]`).

### PyTorch Streaming DataLoader Performance
- **Batching:** Mini-batch size $B=512$, deterministic worker seeding.
- **Memory Footprint:** Only $20.17\,\text{MB}$ RAM for full training tensors.
- **Throughput:** Exceeds $575,000\,\text{samples/sec}$, streaming full training epochs in $<0.30\,\text{s}$.
- **Unsupervised Objective:** Target tensor is identical to clean input: $(\mathbf{x}, \mathbf{x})$.

---

## Slide 3: Anti-Leakage Invariants & Early-Stopping Hygiene

### Five Core Invariant Guarantees
1. **Legit-Only Training Isolation:** All $360$ training fraud cases are strictly excluded; training DataLoader contains strictly $170,524$ legitimate transactions (Class $== 0$).
2. **Frozen Normalization:** `StandardScaler` is fitted *strictly* on training legitimate records; validation and test sets apply frozen $(\boldsymbol{\mu}, \boldsymbol{\sigma})$.
3. **Early-Stopping Loss Hygiene:** Early stopping monitors reconstruction loss strictly on **validation-legitimate transactions** ($56,904$ records, patience $= 10$). Fraud labels *never* influence training epochs.
4. **Zero Test Contamination:** Test set is completely isolated and evaluated only once during final benchmark generation.
5. **Rejection of Synthetic SMOTE:** Fraud is non-stationary and delayed by verification lag; synthesizing artificial fraud points via linear interpolation distorts the anomaly boundary.

---

## Slide 4: Feature Variance Dominance & Reconstruction MSE Scoring

### The Mathematical Problem
- **Coordinate-Isotropic Loss:** Unweighted Mean Squared Error treats all dimensions uniformly:
  $$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{d} \sum_{j=1}^d (x_j - \hat{x}_j)^2$$
- **Variance Imbalance:** In unscaled space, raw transaction `Amount` has variance $\sigma^2 \approx 62,500\,\text{EUR}^2$, whereas PCA components $V_1 \dots V_{28}$ have variances $\le 3.5$.
- **Gradient Monopolization:** Without standard scaling, errors in `Amount` contribute $>99.9\%$ of the gradient, blinding the autoencoder to deviations across the remaining 28 latent features.

### The Solution: Pre-Training Normalization
- **Enforced Scale Uniformity:** Post-scaling variance is normalized to $\sigma^2 = 1.0000 \pm 0.0002$ for all 31 features on training legitimate records.
- **Balanced Gradient Updates:** Each coordinate receives an equal theoretical baseline weight of $1/31 \approx 3.23\%$ in reconstruction error.
- **Preserved Anomaly Contrast:** Abnormal feature excursions in fraud transactions produce sharp, multi-coordinate reconstruction spikes without scale bias.
