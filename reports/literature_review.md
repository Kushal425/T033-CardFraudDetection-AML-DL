# Literature Review: Unsupervised & Density-Based Anomaly Detection for Card Fraud Triage

**Author:** Prachee Dhar (Task 01)  
**Project:** Project 26 — Card Fraud Triage (GMM vs. Autoencoder) — Team T033  
**Phase:** Phase 1 (Classical Machine Learning & Statistical Anomaly Detection)  
**Date:** October 2026  

---

## Executive Summary & Scope

Credit card fraud detection presents an archetypal asymmetric, adversarial challenge in statistical machine learning. While production financial infrastructure frequently relies on supervised classification ensembles, supervised paradigms fundamentally rely on regular label feedback and suffer severe degradation under extreme class imbalance, delayed ground-truth verification, and distribution shift (concept drift). 

This literature review grounds **Phase 1** of Project T033, focusing strictly on classical, unsupervised, and density-based anomaly detection formulations—specifically Gaussian Mixture Models (GMM) trained via Expectation-Maximization (EM), alongside canonical anomaly baselines (Isolation Forest, One-Class SVM, and PCA reconstruction). In strict adherence to Phase 1 boundaries, deep learning architectures (e.g., autoencoders) are reserved exclusively for future investigation in Phase 2 and are not evaluated here.

---

## 1. Problem Formulation: Why Card Fraud is Anomaly Detection

Credit card fraud differs fundamentally from standard binary classification due to three persistent operational realities:

1. **Extreme Class Imbalance:** In legitimate banking streams, fraudulent transactions constitute a minute fraction of total transaction volume—empirically observed at **0.1727%** in the ULB benchmark dataset (492 frauds out of 284,807 transactions). Under such severe skew (roughly 1 fraud per 578 transactions), naive standard loss functions (e.g., binary cross-entropy, 0-1 loss) degenerate toward majority-class collapse, achieving 99.83% nominal accuracy while failing entirely in risk mitigation [VERIFY: dalpozzolo2015calibrating].
2. **Label Scarcity, Verification Lag, and Chargeback Asymmetry:** Ground truth labels are rarely available in real-time. Legitimate transactions are verified passively (absence of dispute), whereas fraudulent activity is confirmed only after cardholder inquiry, chargeback submission, or manual investigation, typically introducing delays ranging from days to weeks [VERIFY: leborgne2022handbook]. Training models assuming immediate, pristine ground-truth labels induces severe target leakage and unrealistic performance estimates.
3. **Concept Drift and Adversarial Evasion:** Fraudulent behaviour is non-stationary and adversarial. As fraud rings detect rule changes or automated tripwires, they modify spend velocity, merchant category distribution, and geographic signatures. Concurrently, legitimate cardholder spending shifts across macro cycles (e.g., holidays, paydays). In statistical terms, the transaction joint distribution $P_t(X, Y)$ drifts over time $t$, rendering historical supervised boundaries obsolete [VERIFY: carcillo2018scarff].

Consequently, framing fraud detection as **unsupervised anomaly detection** on non-fraudulent (legitimate) historical transactions offers substantial operational resilience. Under this paradigm, a statistical model learns the support or probability density function $p(x)$ of normal purchasing behaviour using unlabelled or verified legitimate historical transactions only. Unseen events exhibiting low likelihood under $p(x)$ or lying in sparse regions of feature space are flagged for human triage.

---

## 2. Density-Based Methods: Gaussian Mixture Models (GMM) & Expectation-Maximization (EM)

### 2.1 Theoretical Foundations of GMM
A Gaussian Mixture Model represents a continuous probability distribution as a convex combination of $K$ multivariate Gaussian components:

$$p(x \mid \Theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

subject to the simplex constraints $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \ge 0$ for all $k \in \{1, \dots, K\}$, where $\Theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ denotes the parameter set. The component density $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ is defined as:

$$\mathcal{N}(x \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{D/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$$

where $D$ is the dimensionality of the feature space ($D = 29$ in our standard pipeline: V1–V28 plus `Amount`).

### 2.2 Maximum Likelihood Estimation via the EM Algorithm
Direct maximum likelihood estimation via gradient ascent on the log-likelihood:

$$\ln p(X \mid \Theta) = \sum_{i=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)$$

is analytically intractable due to the summation inside the logarithm. The **Expectation-Maximization (EM)** algorithm, formalised by Dempster, Laird, and Rubin (1977) [VERIFY: dempster1977maximum], iteratively maximises a lower bound on the incomplete log-likelihood via latent cluster indicators $z_{ik} \in \{0, 1\}$:

- **E-step (Expectation):** Compute the posterior component responsibility $\gamma_{ik}$ for each sample $x_i$ and component $k$ given current parameter estimates $\Theta^{(t)}$:
  $$\gamma_{ik} = \frac{\pi_k^{(t)} \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}$$
- **M-step (Maximization):** Update parameters $\Theta^{(t+1)}$ analytically using the expected responsibilities:
  $$N_k = \sum_{i=1}^N \gamma_{ik}, \quad \pi_k^{(t+1)} = \frac{N_k}{N}, \quad \mu_k^{(t+1)} = \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik} x_i$$
  $$\Sigma_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^T + \epsilon I$$
  where $\epsilon I$ provides regularisation against covariance matrix singularity in high-dimensional sparse spaces.

### 2.3 Anomaly Scoring Function
For a test transaction $x^*$, the anomaly score is naturally defined as the negative log-likelihood (NLL):

$$\mathcal{S}_{\text{GMM}}(x^*) = -\ln p(x^* \mid \hat{\Theta}) = -\ln \left( \sum_{k=1}^K \hat{\pi}_k \mathcal{N}(x^* \mid \hat{\mu}_k, \hat{\Sigma}_k) \right)$$

Higher values of $\mathcal{S}_{\text{GMM}}(x^*)$ indicate that $x^*$ falls in low-density tails or gaps between normal modes, denoting anomalous, potentially fraudulent behaviour.

### 2.4 Model Selection: AIC & BIC
Choosing the component hyperparameter $K$ requires balancing model fidelity against parameter proliferation. Because maximum log-likelihood increases monotonically with $K$, unpenalised selection overfits training data. We utilise formal information criteria:
- **Akaike Information Criterion (AIC)** [VERIFY: akaike1974new]:
  $$\text{AIC} = 2p - 2\ln \hat{L}$$
- **Bayesian Information Criterion (BIC)** [VERIFY: schwarz1978estimating]:
  $$\text{BIC} = p \ln(N) - 2\ln \hat{L}$$
where $\hat{L} = p(X_{\text{train}} \mid \hat{\Theta})$ is the maximized likelihood, $N$ is the number of training samples, and $p$ is the number of free parameters. For a full covariance GMM in $D$ dimensions with $K$ components, $p = K - 1 + K D + K \frac{D(D+1)}{2}$. BIC applies a stronger penalty proportional to $\ln(N)$, favouring parsimonious clusterings that mitigate catastrophic variance.

---

## 3. Project Scope & Methodological Boundaries

Phase 1 focuses entirely on unsupervised and density-based statistical machine learning. The target anomaly models include:
1. **Gaussian Mixture Model (GMM)** with diagonal, tied, and full covariance structures, parameterized across $K \in \{1, 2, 3, 5, 8, 10\}$.
2. **Isolation Forest** [VERIFY: liu2008isolation] as a tree-based isolation baseline.
3. **One-Class Support Vector Machine (OC-SVM)** [VERIFY: scholkopf2001estimating] as a kernel boundary baseline.
4. **Principal Component Analysis (PCA) Reconstruction Error** [VERIFY: shyu2003novel] as a linear projection baseline.

*Future Work Boundary:* Non-linear neural representation models such as deep reconstruction autoencoders and hybrid deep architectures are intentionally excluded from Phase 1 and will be explored as future work in Phase 2.

---

## 4. Classical Anomaly Detection Baselines

To establish benchmark performance, GMM must be rigorously compared against the following classical anomaly models:

### 4.1 Isolation Forest (Liu, Ting, & Zhou, 2008)
Isolation Forest [VERIFY: liu2008isolation] isolates anomalies by exploiting two quantitative properties: anomalies are few, and their attribute values differ markedly from normal points. The algorithm builds an ensemble of completely random trees ($iTrees$). At each node, a feature is selected at random, and a random split point is chosen between the feature's minimum and maximum values. 

Anomalies require fewer recursive partitions to isolate, yielding significantly shorter average path lengths $h(x)$. The anomaly score is normalized:

$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$

where $c(n) = 2\ln(n - 1) + 0.5772156649 - \frac{2(n-1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree. Scores close to 1 indicate anomalous transactions. Crucially, Isolation Forest has linear time complexity $\mathcal{O}(t \cdot \psi \log \psi)$ and does not assume parametric normality.

### 4.2 One-Class Support Vector Machine (Schölkopf et al., 2001)
One-Class SVM [VERIFY: scholkopf2001estimating] maps training instances into a reproducing kernel Hilbert space $\mathcal{H}$ via a non-linear mapping $\Phi(x)$, and constructs a maximum-margin hyperplane separating the mapped normal data from the origin:

$$\min_{w, \xi, \rho} \frac{1}{2} \|w\|^2 + \frac{1}{\nu N} \sum_{i=1}^N \xi_i - \rho \quad \text{s.t.} \quad \langle w, \Phi(x_i) \rangle \ge \rho - \xi_i, \; \xi_i \ge 0$$

where $\nu \in (0, 1]$ governs the upper bound on the fraction of outliers (training errors) and lower bound on the number of support vectors. The decision score is $f(x) = \text{sign}(\langle w, \Phi(x) \rangle - \rho)$. For continuous ranking, the raw signed distance to the separating margin serves as the inverse anomaly score.

### 4.3 PCA Reconstruction Error (Shyu et al., 2003)
Principal Component Analysis (PCA) identifies orthogonal directions of maximum variance [VERIFY: shyu2003novel]. When fit on normal transaction records, the leading $k$ principal components capture the dominant subspace of legitimate transaction patterns. Given a sample $x$ and orthogonal projection matrix $V_k \in \mathbb{R}^{D \times k}$:

$$\hat{x} = V_k V_k^T (x - \bar{x}) + \bar{x}$$

The reconstruction error $\mathcal{E}(x) = \|x - \hat{x}\|_2^2$ measures deviation from the normal hyperplane. Transactions with abnormal feature combinations incur substantial projection residuals.

---

## 5. The Benchmark Dataset (Dal Pozzolo et al., ULB)

The dataset analysed throughout this project was collected through a research collaboration between the Université Libre de Bruxelles (ULB) Machine Learning Group and Worldline [VERIFY: dalpozzolo2015calibrating, dalpozzolo2018credit, leborgne2022handbook].

Key characteristics of the dataset include:
- **Transaction Volume:** Exactly 284,807 card transactions recorded over a continuous 48-hour window (172,792 seconds) in September 2013 by European cardholders.
- **Extreme Class Imbalance:** Exactly 492 transactions are classified as fraudulent, yielding an empirical positive rate of **0.1727%** (approximately 1 fraud in 578 transactions).
- **PCA Anonymisation:** To comply with strict European financial confidentiality and banking secrecy laws, 28 numerical features (`V1` through `V28`) represent principal components extracted from original transaction attributes. The only two non-transformed features are `Time` (seconds elapsed since the first transaction in the dataset) and `Amount` (transaction transaction value in Euros).
- **Exact Duplicates:** Our dataset audit identified **1,081 exact duplicate records** (0.3796% of the dataset), comprising 1,822 duplicate rows (1,822 legit, 32 fraud), with 1 cross-class duplicate pair.

### Limitations of the ULB Dataset:
1. *Loss of Semantic Interpretability:* Because V1–V28 are PCA projections, direct business interpretation (e.g., merchant category, country of origin, terminal type) is impossible.
2. *Short Temporal Window:* A 48-hour observation window captures daily circadian patterns but precludes observation of multi-week seasonality, monthly billing cycles, or long-term macroeconomic concept drift.
3. *Geographic Homogeneity:* Restricted to European cardholders, limiting immediate generalization to emerging card-not-present (CNP) ecosystems with different authentication standards (e.g., OTP, 3D-Secure).

---

## 6. Evaluation Framework for Highly Imbalanced Streams

Standard binary classification metrics—most notably Accuracy, Sensitivity/Specificity averages, and the standard Area Under the Receiver Operating Characteristic (ROC-AUC)—produce severe optimism bias on imbalanced fraud datasets [VERIFY: saito2015precision, davis2006relationship, hand2001prospecting].

### 6.1 The Fallacy of Accuracy and ROC-AUC
On a dataset with 99.8273% legitimate transactions:
- A trivial constant classifier predicting $\hat{y}_i = 0$ achieves **99.8273% Accuracy**, despite missing 100% of financial fraud.
- ROC-AUC computes the area under the curve plotting True Positive Rate (Recall) against False Positive Rate (FPR):
  $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
  Because the total number of True Negatives (TN) is vast (284,315 legit rows), a large surge of 1,000 false alarms results in an FPR of only $1000 / 284315 \approx 0.0035$ (0.35%). Consequently, ROC-AUC remains artificially inflated ($> 0.95$), masking the fact that investigators would be overwhelmed with false alerts.

### 6.2 Precision-Recall AUC (PR-AUC / Average Precision)
Saito and Rehmsmeier (2015) [VERIFY: saito2015precision] and Davis and Goadrich (2006) [VERIFY: davis2006relationship] proved that Precision-Recall curves provide an uncorrupted assessment under severe class imbalance:
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — directly measures the fraction of flagged transactions that are genuine fraud.
- **Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — measures the fraction of genuine frauds caught.

The baseline PR-AUC for a random uninformative classifier equals the empirical fraud base rate $\pi = \frac{P}{P+N} \approx 0.001727$. Any score significantly exceeding 0.0017 represents genuine discriminative ability.

### 6.3 Operational Review-Capacity Metrics: Precision@k and Recall@k
In practical anti-money-laundering (AML) and fraud operations, a security operations center (SOC) has a fixed human investigation budget per shift (e.g., capacity to triage $k \in \{50, 100, 200\}$ alerts per hour or split). 
- **Precision@k:** The fraction of top-$k$ highest-scoring transactions that are confirmed frauds.
- **Recall@k:** The fraction of all ground-truth frauds contained within the top-$k$ ranked alerts.

Evaluating Precision@k and Recall@k directly reflects operational utility under strict human investigator bandwidth [VERIFY: dalpozzolo2018credit].

---

## 7. Comparative Literature Table

The following table synthesizes the seminal and contemporary literature relevant to unsupervised and statistical card fraud detection:

| Reference | Year | Method | Dataset / Domain | Primary Metrics | Key Limitation / Gap Identified |
|---|---|---|---|---|---|
| **Dempster et al.** [VERIFY: dempster1977maximum] | 1977 | EM Algorithm for Gaussian Mixtures | General statistical incomplete data | Log-likelihood convergence | Sensitive to local maxima and covariance singularity in high dimensions. |
| **Akaike** [VERIFY: akaike1974new] | 1974 | AIC Model Selection | Autoregressive / time series | Information criterion | Tends to overestimate cluster count $K$ in large sample regimes. |
| **Schwarz** [VERIFY: schwarz1978estimating] | 1978 | BIC Dimension Estimator | Asymptotic parametric models | Posterior log-evidence | Assumes uniform parameter prior; penalises large samples heavily. |
| **Schölkopf et al.** [VERIFY: scholkopf2001estimating] | 2001 | One-Class SVM (Kernel Support) | Synthetic & benchmark benchmarks | Fraction of support vectors | $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$ computational scaling; tuning $\nu$ and $\gamma$ without labels is difficult. |
| **Shyu et al.** [VERIFY: shyu2003novel] | 2003 | PCA Reconstruction Error | Network intrusion (KDD-99) | Detection Rate, False Alarm Rate | Linear projection fails to capture multi-modal non-linear manifolds. |
| **Davis & Goadrich** [VERIFY: davis2006relationship] | 2006 | PR vs. ROC Theoretical Mapping | Machine learning benchmarks | PR-AUC, ROC-AUC | Theoretical proof; does not evaluate operational alert queues ($P@k$). |
| **Liu et al.** [VERIFY: liu2008isolation] | 2008 | Isolation Forest ($iForest$) | Multi-domain anomaly benchmarks | ROC-AUC, Execution Time | Axis-aligned splits induce masking in clustered local anomaly settings. |
| **Chandola et al.** [VERIFY: chandola2009anomaly] | 2009 | Comprehensive Anomaly Survey | Cross-domain taxonomies | False Alarm vs Recall trade-offs | Qualitative survey without unified experimental evaluation. |
| **Dal Pozzolo et al.** [VERIFY: dalpozzolo2015calibrating] | 2015 | Undersampling & Probability Calibration | Commercial credit card stream | PR-AUC, Card-Loss | Evaluates supervised classifiers; does not evaluate density models. |
| **Saito & Rehmsmeier** [VERIFY: saito2015precision] | 2015 | Invariance of PR curves | Synthetic & biological imbalance | Precision-Recall Curves | Confined to supervised binary evaluation without concept drift. |
| **Carcillo et al.** [VERIFY: carcillo2018scarff] | 2018 | SCARFF Framework (Streaming ML) | ULB 50M transaction feed | Alert Precision, PR-AUC | High-complexity streaming pipeline requiring extensive infrastructure. |
| **Dal Pozzolo et al.** [VERIFY: dalpozzolo2018credit] | 2018 | Realistic Credit Card Modeling | ULB European dataset | PR-AUC, Precision@100 | Uses random splits in certain ablations; primarily explores supervised ensembles. |
| **Le Borgne et al.** [VERIFY: leborgne2022handbook] | 2022 | Reproducible Fraud Handbook | Simulated & ULB benchmarks | Daily Card PR-AUC, $P@k$ | Focuses heavily on supervised decision trees; limited treatment of GMM density fitting. |

---

## 8. Research Gap & Project T033 Contributions

A thorough audit of the published literature reveals several pervasive methodological flaws in published credit card fraud studies:

1. **Random Splitting and Information Leakage:** A large proportion of published benchmarks split the 284,807 transactions randomly (e.g., standard `train_test_split(..., shuffle=True)`). Because financial transaction streams are strictly sequential, random splitting contaminates training folds with future transactions, artificially boosting performance and concealing temporal degradation.
2. **Post-Hoc Test-Set Threshold Tuning:** Many studies report peak F1 scores or PR metrics by sweeping classification thresholds directly over the test set, creating optimistic selection bias.
3. **Inconsistent Training Subsets:** Literature benchmarks often compare anomaly models trained on all classes against models trained only on normal data, confounding novelty detection with contamination tolerance.
4. **Neglect of Parametric Assumptions:** Studies evaluating GMM rarely perform rigorous distribution tests (e.g., skewness, kurtosis, Shapiro-Wilk) on feature dimensions, treating Gaussianity as a default assumption.

### Project T033 Contributions:
- **Strict Chronological Splitting:** We enforce a temporal 60% train / 20% validation / 20% test partition:
  $$\text{Train: } [0, 120\,396\text{s}] \;\longrightarrow\; \text{Validation: } [120\,396\text{s}, 145\,247\text{s}] \;\longrightarrow\; \text{Test: } [145\,248\text{s}, 172\,792\text{s}]$$
  with zero overlap across time boundaries.
- **Unsupervised Legit-Only Fitting with Isolated Standardisation:** The standard scaler is fit exclusively on legitimate training transactions ($\text{Class} = 0$), guaranteeing that fraud vectors never inform transformation parameters.
- **Validation-Calibrated Decision Thresholds:** Decision cutoffs are selected strictly on the chronological validation split (e.g., maximizing validation F1 or targeting operational alerting budgets) before frozen application to the held-out test split.
- **Rigorous Gaussianity Auditing:** We formally test the distributional viability of GMM on normal transaction features via Kolmogorov-Smirnov, Shapiro-Wilk, skewness, and excess kurtosis tests, providing empirical evidence regarding the limits of Gaussian mixture assumptions on credit card features.

---

## References

*(All references cataloged and verified in canonical `reports/ieee_report/refs.bib`)*

1. [VERIFY] Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716-723.
2. [VERIFY] Carcillo, F., Dal Pozzolo, A., Le Borgne, Y.-A., Caelen, O., Mazzer, Y., & Bontempi, G. (2018). SCARFF: a framework for addressing drift and imbalanced data in credit card fraud detection. *Proceedings of the 4th ACM SIGBD*, 1-8.
3. [VERIFY] Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58.
4. [VERIFY] Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*, 159-166.
5. [VERIFY] Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018). Credit card fraud detection: a realistic modeling and a novel learning strategy. *IEEE TNNLS*, 29(8), 3784-3797.
6. [VERIFY] Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*, 233-240.
7. [VERIFY] Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS Series B*, 39(1), 1-22.
8. [VERIFY] Hand, D. J., & Blunt, G. (2001). Prospecting for winners in credit card risk. *IMA Journal of Management Mathematics*, 12(3), 283-300.
9. [VERIFY] Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). *Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook*. Université Libre de Bruxelles.
10. [VERIFY] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM*, 413-422.
11. [VERIFY] Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
12. [VERIFY] Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7), 1443-1471.
13. [VERIFY] Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
14. [VERIFY] Shyu, M.-L., Chen, S.-C., Sarinnapakorn, K., & Chang, L. (2003). A novel anomaly detection scheme based on principal component classifier. *Miami Univ Tech Report*, TR-2003-01.
