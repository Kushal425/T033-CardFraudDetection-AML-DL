# Phase 2 Master Viva Defense Guide: Deep Learning Literature & Data Pipeline

**Author:** Krishna Pramod Dubale (Enrollment: 2401010239)  
**Role:** Deep Learning Literature Review, DL Data Pipeline Documentation & Invariant Auditing  
**Team:** T033, Project 26 — Card Fraud Triage  
**Deliverable:** `docs/viva_notes/krishna_p2.md`  

---

## 1. High-Yield Viva Questions & Answers

### Q1: Why do we train our Deep Autoencoder strictly on legitimate transactions (legit-only training)?
**Answer:**
There are three fundamental reasons:
1. **Verification Lag in Production Banking:** In real financial payment systems, fraudulent transactions cannot be reliably labeled at authorization time. Ground-truth confirmation requires 30 to 60 days of dispute resolution, cardholder inquiries, and chargeback processing (Dal Pozzolo et al., 2018 `[VERIFY]`). Training supervised models requires waiting for these delayed labels, creating massive vulnerability to novel fraud patterns. An unsupervised anomaly detector trained exclusively on unflagged, legitimate transactions can be deployed and continuously retrained without waiting for chargeback labels.
2. **One-Class Normality Modeling:** The objective of an undercomplete autoencoder is to learn the geometric structure and low-dimensional manifold of *normal consumer spending*. If fraudulent transactions were included in training, the network's high representational capacity would allocate weights toward reconstructing fraud patterns, directly undermining its ability to flag fraud as an anomaly.
3. **Severe Class Imbalance ($0.17\%$ base rate):** In a dataset with 1 fraud per 578 transactions, supervised gradients are overwhelmingly dominated by legitimate examples. Rather than artificially distorting the training distribution with oversampling (e.g. SMOTE), the one-class formulation naturally circumvents class imbalance by design.

---

### Q2: Why does validation-legit early stopping not leak fraud labels, whereas standard validation early stopping does?
**Answer:**
Early stopping is an active algorithmic regularizer: it dictates the exact optimization stopping point and selects the specific parameter checkpoint $\boldsymbol{\theta}^* = \arg\min_e \mathcal{L}_{\text{val}}(e)$.
- **The Leakage Mechanism in Standard Early Stopping:** If an engineer monitors validation PR-AUC, F1-score, or a loss function evaluated over a mixed dataset containing fraud labels, the choice of stopping epoch directly uses ground-truth fraud labels. This feedback signal acts as covert supervision, leaking test-relevant label information into the training phase.
- **Our Leakage-Free Design:** We construct a dedicated validation-legitimate tensor $\mathbf{X}_{\text{val,legit}}$ containing strictly the $56,904$ legitimate transactions of the validation split (excluding all $57$ validation fraud cases). Early stopping monitors:
  $$\mathcal{L}_{\text{val,legit}}(e) = \frac{1}{|\mathcal{V}_{\text{legit}}|} \sum_{\mathbf{x} \in \mathcal{V}_{\text{legit}}} \|\mathbf{x} - g_{\boldsymbol{\phi}}(f_{\boldsymbol{\theta}}(\mathbf{x}))\|_2^2$$
  Because zero fraud labels or fraud feature vectors are evaluated during early-stopping checks, the training convergence remains $100\%$ unsupervised. The $57$ validation fraud cases are reserved exclusively for post-hoc threshold selection and calibration.

---

### Q3: Why do high-variance features dominate Mean Squared Error (MSE) reconstruction error, and what is the mathematical fix?
**Answer:**
The autoencoder Mean Squared Error loss is coordinate-isotropic (an unweighted sum over all coordinates):
$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{d} \sum_{j=1}^d (x_j - \hat{x}_j)^2$$
Under gradient descent, the gradient with respect to the output layer weights $\mathbf{W}_d$ for coordinate $j$ is:
$$\frac{\partial \mathcal{L}}{\partial \hat{x}_j} = -\frac{2}{d} (x_j - \hat{x}_j)$$
If coordinate $j$ has a large variance $\sigma_j^2 \gg \sigma_i^2$, the expected residual error $\mathbb{E}[(X_j - \hat{X}_j)^2]$ is proportional to $\sigma_j^2$.
- **In our dataset:** Raw transaction `Amount` has an empirical variance of $\approx 62,500\,\text{EUR}^2$, whereas individual PCA components $V_1 \dots V_{28}$ have variances between $0.1$ and $3.5$. Without normalization, errors in `Amount` contribute $>99.9\%$ of the gradient magnitude. Backpropagation focuses almost all capacity on reconstructing `Amount`, completely ignoring subtle adversarial distortions across $V_1 \dots V_{28}$.
- **The Mathematical Fix:** We apply `StandardScaler` fitted strictly on training legitimate records:
  $$z_j = \frac{x_j - \mu_j^{\text{train,legit}}}{\sigma_j^{\text{train,legit}}}$$
  This normalizes $\operatorname{Var}(Z_j) = 1.0000 \pm 0.0002$ across all coordinates, ensuring that each feature contributes an equal theoretical baseline share of $1/d \approx 3.23\%$ to the reconstruction loss.

---

### Q4: Why is SMOTE or synthetic oversampling mathematically flawed for unsupervised fraud triage?
**Answer:**
Synthetic Minority Oversampling Technique (SMOTE) generates synthetic minority instances by finding the $k$-nearest neighbors of a fraud point and interpolating:
$$\tilde{\mathbf{x}} = \mathbf{x}_i + \lambda (\mathbf{x}_{k} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
This is fundamentally flawed in card fraud triage for three reasons:
1. **Convexity Assumption Violation:** SMOTE assumes that the minority class distribution is convex and continuous. In reality, credit card fraud is adversarial, heterogeneous, and disjoint: a fraudulent transaction in electronics purchasing shares no continuum with a fraudulent transaction in micro-billing. Linear interpolation places synthetic frauds in non-fraudulent consumer spending regions, corrupting decision boundaries.
2. **Incompatibility with One-Class Paradigms:** In unsupervised anomaly detection, the model never trains on fraud. Introducing synthetic frauds directly violates the one-class premise.
3. **Data Leakage & Non-Stationarity:** SMOTE uses existing fraud labels. In production, label delay means past fraud signatures are outdated; training on interpolated past frauds creates strong overfitting to stale attack vectors (Lucas et al., 2020 `[VERIFY]`).

---

### Q5: What is the exact literature gap our Phase 2 architecture addresses?
**Answer:**
Existing literature exhibits a clear divide:
1. **Pure Reconstruction Models (AE / PCA):** Measure Euclidean orthogonal distance to the learned manifold ($\|\mathbf{x} - \hat{\mathbf{x}}\|^2$). While effective at catching off-manifold outliers (as PCA proved in Phase 1 with $\text{PR-AUC}=0.7425$), they are completely blind to probability density *on* the manifold. A fraudulent transaction that lies within the nominal subspace but in an ultra-low density region achieves low reconstruction error and escapes detection.
2. **Pure Density Models (GMM):** Estimate full-space multivariate densities ($p(\mathbf{x})$). However, full-covariance GMM in 30 dimensions requires estimating $\frac{d(d+1)}{2} = 465$ covariance parameters per mixture component, suffering from severe curse of dimensionality and covariance singularity collapses.
3. **End-to-End Joint Models (DAGMM):** Zong et al. (2018) `[VERIFY]` proposed joint optimization, but end-to-end training is notoriously unstable, prone to EM covariance collapse and sensitive to singular penalty weights.

**Our Phase 2 Solution (The Decoupled Hybrid):**
We decouple the representation learning from the density estimation:
- **Undercomplete AE:** Compresses the 30D transaction space to an 8D bottleneck manifold $\mathbf{z} \in \mathbb{R}^8$.
- **Latent GMM ($H_1$):** Estimates density $p(\mathbf{z})$ in the compact 8D space, reducing covariance parameters from 465 to $\frac{8 \times 9}{2} = 36$ per component (a 92% reduction).
- **Hybrid Fusion ($H_2$) and DAGMM-Lite ($H_3$):** Fuses orthogonal signals: off-manifold distance (reconstruction error) and on-manifold density ($-\log p(\mathbf{z})$), achieving superior robustness without joint training instability.

---

## 2. Comprehensive Reference Matrix & Contribution Summary

| Citation | Venue / Year | Key Contribution | Role in Team T033 Phase 2 |
|---|---|---|---|
| **Hinton & Salakhutdinov** `[VERIFY]` | *Science* 2006 | Dimensionality reduction via deep multilayer autoencoders | Theoretical foundation for replacing linear PCA with non-linear undercomplete autoencoders. |
| **Vincent et al.** `[VERIFY]` | *ICML* 2008 | Denoising Autoencoder (DAE) with input noise corruption | Inductive bias preventing identity mapping; robustness against transaction jitter. |
| **Vincent** `[VERIFY]` | *Neural Comput.* 2011 | Connection between DAEs and Score Matching | Proves reconstruction vector points along the gradient of data log-density $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. |
| **Sakurada & Yairi** `[VERIFY]` | *MLSDA* 2014 | Anomaly detection using autoencoder reconstruction error | Standardized the reconstruction MSE metric $s(\mathbf{x}) = \frac{1}{d}\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ as an anomaly score. |
| **An & Cho** `[VERIFY]` | *SNU Tech Rep.* 2015 | Variational Autoencoders for anomaly detection | Grounding for reconstruction probability vs. deterministic MSE. |
| **Zong et al.** `[VERIFY]` | *ICLR* 2018 | Deep Autoencoding GMM (DAGMM) | Inspiration for hybrid models combining latent codes $\mathbf{z}$ and reconstruction distance. |
| **Kingma & Ba** `[VERIFY]` | *ICLR* 2015 | Adam optimizer (adaptive moment estimation) | Robust gradient descent avoiding saddle points in autoencoder bottlenecks. |
| **Srivastava et al.** `[VERIFY]` | *JMLR* 2014 | Dropout regularization | Regularizer preventing co-adaptation of weights and over-generalization. |
| **Ioffe & Szegedy** `[VERIFY]` | *ICML* 2015 | Batch Normalization | Invariant analysis showing why running stats must freeze training legit statistics. |
| **Goodfellow et al.** `[VERIFY]` | *MIT Press* 2016 | Deep Learning textbook (early stopping & regularization) | Formal justification of validation-legit early stopping as an implicit $L_2$ regularizer. |
| **Gong et al.** `[VERIFY]` | *ICCV* 2019 | Memory-augmented autoencoders (MemAE) | Rigorous analysis of the over-generalization pathology in autoencoders. |
| **Pumsirirat & Yan** `[VERIFY]` | *IJMLC* 2018 | Autoencoders & RBM for card fraud detection | Empirical benchmark comparison on ULB dataset; highlights metric pitfalls. |
| **Roy et al.** `[VERIFY]` | *Syst. Sci. Control* 2018 | Deep learning for financial fraud detection | Evidence that deep representations extract robust features from noisy financial streams. |
| **Lucas et al.** `[VERIFY]` | *Mach. Learn.* 2020 | Dataset shift and temporal evaluation in fraud streams | Methodological proof that chronological splitting is mandatory to prevent look-ahead leakage. |
| **Dal Pozzolo et al.** `[VERIFY]` | *IEEE TNNLS* 2018 | Realistic modeling and benchmark evaluation | Origin of the ULB benchmark; defines verification lag and operational constraints. |

---

## 3. Data Pipeline Implementation Summary

- **Script:** `src/eda_dl.py`
- **Notebook:** `notebooks/07_dl_data_checks.ipynb`
- **Unit Tests:** `tests/test_dl_data.py` (8 passing unit tests)
- **Tensors Created:**
  - `X_train_legit`: `(170524, 31)`, `float32`, $0$ fraud cases.
  - `X_val_legit`: `(56904, 31)`, `float32`, $0$ fraud cases (early stopping).
  - `X_val`: `(56961, 31)`, `float32`, $57$ fraud cases.
  - `X_test`: `(56962, 31)`, `float32`, $75$ fraud cases.
- **DataLoader Throughput:** $>575,000\,\text{samples/sec}$, batch size $512$, $20.17\,\text{MB}$ memory footprint.
- **Audit Result:** $100\%$ PASS across all anti-leakage and scaling checks.
