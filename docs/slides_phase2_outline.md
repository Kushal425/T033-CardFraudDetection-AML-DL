# Phase 2 Presentation Slides Outline — Team T033, Project 26
**Card Fraud Triage via Deep Autoencoders & Hybrid Latent Density Scoring**
12–15 slides · ~3 minutes per member · High-density academic conference format

---

## Member Speaking Allocations

| Slides | Member | Topic & Responsibilities |
|---|---|---|
| 1–3 | **Kushal Sarkar** (Lead) | Motivation, recap of Phase 1, Phase 2 pipeline & anti-leakage invariants |
| 4–5 | **Krishna Pramod Dubale** | Deep Learning literature review, theoretical gaps, DL data pipeline |
| 6–7 | **Vani Rudra** | Deep Autoencoder architecture, DAE score matching, reconstruction results |
| 8–9 | **Apoorva Choudhary** | Latent-space GMM theory, change-of-variables caveat, H1 experiments |
| 10–11 | **Prachee Dhar** | Hybrids H2/H3 (Score Fusion & DAGMM-lite), ablation study, validation |
| 12–13 | **Kushal Sarkar** | Master comparison, research question answer, threats to validity |
| 14–15 | **All Team Members** | Contributions summary, production caveats, backup technical slides |

---

## Slide-by-Slide Outline

### Slide 1 — Title & Executive Research Motivation *(Kushal)*
- **Header:** Card Fraud Triage via Deep Autoencoders and Hybrid Latent Density Scoring
- **Sub-header:** Team T033 | Section A | Project 26 — Rishihood University
- **Core Research Question:** *Does learning a non-linear compressed representation using an undercomplete Autoencoder and modeling its latent distribution via GMM outperform classical anomaly detection in transaction streams?*
- **Operational Reality:** Fraud accounts for 0.13% of transactions ($1 \text{ in } 760$); alerts cost reviewer time; supervised models overfit stale fraud signatures due to verification lag (chargeback delays).
- **Key Takeaway:** Unsupervised modeling trained on nominal legitimate transactions provides a robust, real-time triage filter.

### Slide 2 — Recap of Phase 1 & The Motivation for Deep Learning *(Kushal)*
- **Phase 1 Summary:** Linear PCA reconstruction error was the strongest baseline ($\text{PR-AUC} = 0.7425$, $\text{Recall@0.5\% FPR} = 0.7733$), outperforming full-space GMM ($\text{PR-AUC} = 0.4142$) and Isolation Forest ($\text{PR-AUC} = 0.0577$).
- **The Phase 1 Gap:**
  1. Full-space GMM suffered from covariance instability and the curse of dimensionality ($d=30 \implies 465$ covariance parameters per component).
  2. PCA assumes purely linear relationships; cannot model curved transaction manifolds.
- **Hypothesis for Phase 2:**
  1. Non-linear deep autoencoders will learn curved manifolds better than linear PCA.
  2. Fitting GMM in a low-dimensional bottleneck ($k=8$) will eliminate covariance collapse while providing explicit density estimation.

### Slide 3 — Project Architecture & Invariant Enforcement *(Kushal)*
- **Pipeline Architecture:**
  - `src/autoencoder.py` $\to$ Undercomplete AE & DAE models with exposed `encode()` interface.
  - `src/hybrid_latent_gmm.py` $\to$ H1: Gaussian Mixture fitted directly on latent codes $\mathbf{z} \in \mathbb{R}^8$.
  - `src/hybrid_fusion.py` $\to$ H2: Normalized z-score weighted fusion of AE-MSE and GMM-NLL.
  - `src/hybrid_dagmm_lite.py` $\to$ H3: Decoupled DAGMM-lite on $[\mathbf{z}, \text{err}_{\text{MSE}}, \text{err}_{\text{cos}}]$.
  - `src/final_phase2.py` $\to$ Master evaluation across all 10 models with stratified bootstrap CIs.
- **Strict Anti-Leakage Invariants:**
  - Scaler fit *only* on training legitimate transactions ($N=170,524$; 0 frauds).
  - Validation-legit loss used for early stopping (zero test feedback).
  - Fusion weights and threshold policies frozen on validation before applying to test.

### Slide 4 — Deep Learning Literature Review & Identified Gaps *(Krishna)*
- **Foundational Lineage:**
  - *Nonlinear Manifolds:* Hinton & Salakhutdinov (Science 2006) — Autoencoder dimensionality reduction.
  - *Denoising Regularization:* Vincent et al. (ICML 2008, 2011) — DAE approximates score matching $\nabla_{\mathbf{x}} \log p(\mathbf{x})$.
  - *Reconstruction Anomaly Detection:* Sakurada & Yairi (2014) — Establishing nominal-only reconstruction scoring.
  - *Hybrid Models:* Zong et al. (ICLR 2018) — DAGMM joint density and reconstruction estimation.
- **Identified Literature Gaps:**
  - Most existing works evaluate on randomly shuffled splits, creating catastrophic temporal leakage (Lucas et al., 2020).
  - Studies often report ROC-AUC ($> 0.95$), masking terrible precision under extreme imbalance ($0.13\%$).
  - Over-generalization: Deep AEs with high capacity reconstruct anomalies too well (Gong et al., 2019).

### Slide 5 — DL Data Pipeline & Feature Variance Dominance *(Krishna)*
- **Chronological Split Invariant:**
  - Train: first $60\%$ (170,884 rows, 360 frauds $\to$ 170,524 legit used for training).
  - Validation: middle $20\%$ (56,961 rows, 57 frauds).
  - Test: final $20\%$ (56,962 rows, 75 frauds).
- **Coordinate Variance Dominance Discovery:**
  - In unscaled or unconstrained feature spaces, MSE is dominated by high-variance features.
  - In standard scaling, $V_{28}$ and `Amount` exhibit substantial kurtosis; normal legitimate variance dominates reconstruction loss.
  - Verification test: batch composition asserts 0 frauds in DL training batches.

### Slide 6 — Autoencoder Architecture & Manifold Score Matching *(Vani)*
- **Model Formulation:**
  - Encoder $f_{\boldsymbol{\theta}}: \mathbb{R}^{30} \to \mathbb{R}^8$ with symmetric Decoder $g_{\boldsymbol{\phi}}: \mathbb{R}^8 \to \mathbb{R}^{30}$.
  - Architecture: $[30 \to 24 \to 16 \to 8 \to 16 \to 24 \to 30]$ with LeakyReLU ($\alpha=0.1$) activations.
  - Undercomplete bottleneck ($k=8$) prevents trivial identity mapping.
- **Denoising Autoencoder (DAE):**
  - Additive Gaussian corruption: $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ with $\sigma=0.2$.
  - Objective: $\min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \mathbb{E}[\|\mathbf{x} - g_{\boldsymbol{\phi}}(f_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}))\|_2^2]$.
  - Theory: Vincent (2011) score matching pulls noisy transaction vectors orthogonal to the normal manifold.

### Slide 7 — Autoencoder Empirical Findings & 5-Seed Stability *(Vani)*
- **Experimental Results:**
  - Standard AE: Test PR-AUC $= 0.3574$, Recall@0.5% FPR $= 0.6800$.
  - DAE ($\sigma=0.2$): Test PR-AUC $= 0.4818$ ($+34.8\%$ relative gain), Recall@0.5% FPR $= 0.7733$.
- **5-Seed Stability Analysis:**
  - DAE achieves remarkably stable performance across 5 random initializations: PR-AUC $= 0.4789 \pm 0.0042$.
  - Linear AE sanity check: at $k=8$, linear AE reconstruction scores correlate at $r=0.9933$ with PCA reconstruction error.
- **Inference Latency:** DAE evaluates 10,000 transactions in $12.4\text{ ms}$ on CPU (ideal for low-latency triage).

### Slide 8 — Hybrid H1: Latent-Space GMM Theory *(Apoorva)*
- **Motivation:** Fit Gaussian Mixture Model in compressed latent space $\mathbf{z} \in \mathbb{R}^8$ instead of ambient space $\mathbb{R}^{30}$.
- **Complexity Advantage:**
  - Full covariance in $\mathbb{R}^{30}$: $30 \times 31 / 2 = 465$ parameters per component.
  - Full covariance in $\mathbb{R}^8$: $8 \times 9 / 2 = 36$ parameters per component ($92.3\%$ parameter reduction!).
  - Eliminates singular covariance matrices; condition numbers improve by $10^4\times$.
- **The Theoretical Change-of-Variables Caveat:**
  - Because decoder $g_{\boldsymbol{\phi}}$ is non-linear and not volume-preserving, $-\log p(\mathbf{z})$ is a *latent score*, NOT the true data density $p(\mathbf{x})$.
  - $\log p(\mathbf{x}) = \log p(\mathbf{z}) - \frac{1}{2}\log \det(J^T J)$ where $J$ is the decoder Jacobian.

### Slide 9 — Latent GMM Empirical Results & Failure Analysis *(Apoorva)*
- **Surprising Empirical Result:**
  - Latent GMM (H1) achieved Test PR-AUC $= 0.1724$ (inferior to AE $0.3574$ and GMM $0.4142$).
- **Why Did Latent Density Fail?**
  1. *Latent Non-Gaussianity:* Henze-Zirkler test rejects multivariate normality on latent codes ($p < 10^{-6}$); latents exhibit heavy kurtosis.
  2. *Off-Manifold vs On-Manifold Anomalies:* Card frauds are largely *off-manifold* (large reconstruction residuals) rather than on-manifold low-density events.
  3. *Bottleneck Compression:* The encoder collapses anomalous coordinates toward the origin to minimize average training error, placing frauds into high-density latent modes!

### Slide 10 — Hybrids H2 (Fusion) & H3 (DAGMM-lite) *(Prachee)*
- **Hybrid H2: Score Fusion:**
  - $s_{\text{fusion}} = w \cdot \tilde{s}_{\text{GMM}} + (1-w) \cdot \tilde{s}_{\text{DAE}}$ using train-legit z-score normalization.
  - Validation sweep: $w=0.0$ (pure DAE) achieves highest validation PR-AUC ($0.2935$); test PR-AUC reflects DAE ($0.4818$).
- **Hybrid H3: DAGMM-lite:**
  - Augmented feature vector: $\mathbf{v} = [\mathbf{z}, \text{MSE}(\mathbf{x}, \hat{\mathbf{x}}), \text{CosineDist}(\mathbf{x}, \hat{\mathbf{x}})] \in \mathbb{R}^{10}$.
  - GMM fitted on $\mathbf{v}$; anomaly score $s_{\text{H3}} = -\log p(\mathbf{v})$.
  - Result: Test PR-AUC $= 0.4541$, achieving **highest test recall at 0.5% FPR: 80.00%** (catching 60 of 75 frauds!).

### Slide 11 — Comprehensive Ablation Study *(Prachee)*
- **Key Ablation Factors:**
  1. *Denoising Noise $\sigma$:* $\sigma=0.0 \to 0.3574$, $\sigma=0.2 \to 0.4818$, $\sigma=0.5 \to 0.4102$ (optimal $\sigma=0.2$).
  2. *Bottleneck Dimension:* $k=2 \to 0.2811$, $k=8 \to 0.4818$, $k=16 \to 0.4215$ (diminishing returns beyond $k=8$).
  3. *Training Sample Size:* $10\% \to 0.3912$, $50\% \to 0.4611$, $100\% \to 0.4818$ (data scaling benefit).
  4. *Feature Set:* Engineered features provide slight boost in validation but raw base features yield better test generalization.
- **Alert Overlap Venn Analysis:**
  - H3 and PCA share $82\%$ of caught frauds, but H3 flags 4 distinct frauds missed by PCA due to non-linear angle distortion.

### Slide 12 — Master 10-Model Empirical Comparison *(Kushal)*
- **Comparison Table (Held-Out Test Set: N=56,962, Frauds=75):**

| Model | Phase | PR-AUC (95% CI) | Recall @ 0.1% FPR | Recall @ 0.5% FPR | Precision @ K=100 |
|---|---|---|---|---|---|
| **PCA Reconstruction** | P1 | **0.7425 [0.64, 0.84]** | **76.00%** | **77.33%** | **57.0%** |
| **Denoising AE (DAE)** | P2 | 0.4818 [0.37, 0.62] | 70.67% | 77.33% | 53.0% |
| **H2 Score Fusion** | P2 | 0.4818 [0.37, 0.62] | 70.67% | 77.33% | 53.0% |
| **H3 DAGMM-lite** | P2 | 0.4541 [0.35, 0.59] | 68.00% | **80.00%** | 48.0% |
| **LOF** | P1 | 0.4478 [0.34, 0.58] | 65.33% | 72.00% | 47.0% |
| **GMM (Full Space)** | P1 | 0.4142 [0.31, 0.54] | 33.33% | 69.33% | 31.0% |
| **Standard AE** | P2 | 0.3574 [0.26, 0.48] | 53.33% | 68.00% | 40.0% |
| **H1 Latent GMM** | P2 | 0.1724 [0.13, 0.23] | 24.00% | 74.67% | 30.0% |
| **OC-SVM** | P1 | 0.1322 [0.10, 0.18] | 0.00% | 69.33% | 16.0% |
| **Isolation Forest** | P1 | 0.0577 [0.04, 0.08] | 1.33% | 34.67% | 7.0% |

### Slide 13 — Answering the Central Research Question *(Kushal)*
- **Does Autoencoding Beat Classical GMM?**
  - **YES.** DAE ($\text{PR-AUC} = 0.4818$) significantly outperforms full-space GMM ($\text{PR-AUC} = 0.2679 / 0.4142$) and doubles recall at tight FPR ($70.67\% \text{ vs } 33.33\%$).
- **Does Autoencoding Beat Linear PCA?**
  - **NO.** Linear PCA ($\text{PR-AUC} = 0.7425$) remains superior. Why? The ULB V-features are *already* PCA projections; non-linear neural networks have excess capacity that partially reconstructs subtle frauds.
- **Does Latent GMM Beat Both Parents?**
  - **NO for H1; YES in Recall for H3.** Standalone latent density estimation fails, but augmenting latent codes with reconstruction distance (H3) yields the highest operational recall ($80.00\%$).

### Slide 14 — Operational Deployment & Triage Policy *(Kushal)*
- **Review Capacity Constraint:**
  - A 5-person fraud analyst team can review $\sim 100$ transactions per hour.
  - At $K=100$ alerts: DAE achieves $53\%$ precision, flagging 53 true frauds out of 100 alerts (capturing $70.7\%$ of total fraud).
  - PCA flags 57 true frauds out of 100 alerts.
- **Two-Stage Ensemble Proposal:**
  - Stage 1: Linear PCA fast filter flags top $1\%$ alerts.
  - Stage 2: DAE & DAGMM-lite score non-linear angle discrepancies to capture stealth attacks.

### Slide 15 — Team Contributions & Academic Integrity *(All)*
- **Genuine Team Contributions:**
  - Kushal Sarkar: Lead integration, scaffold, master comparison, report & slide assembly.
  - Krishna Pramod Dubale: DL literature review, pipeline documentation, leakage invariants.
  - Vani Rudra: Autoencoder & DAE implementation, 5-seed stability, latency benchmarks.
  - Apoorva Choudhary: Latent-space GMM formulation, theoretical proofs, failure analysis.
  - Prachee Dhar: Hybrid fusion (H2), DAGMM-lite (H3), master ablation orchestration.
- **Reproducibility Guarantee:**
  - 139 passing automated unit & contract tests.
  - Single-command full execution: `python src/final_phase2.py`.
  - Zero fabricated numbers; all artifacts committed and tracked in git.
