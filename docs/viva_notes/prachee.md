# Viva Notes — Prachee Dhar (Literature Review, EDA & Dataset Quality)
**Task 01 — P1_01 — Team T033, Project 26**
**Phase 1 (Machine Learning & Statistical Anomaly Detection Only)**

This document explains every plot, table, decision, and theoretical concept in plain English so I can answer questions confidently in my individual viva without memorizing code.

---

## 1. Core Viva Concept Q&A (The Big Six)

### Q1: Why PR-AUC instead of Accuracy or ROC-AUC?
- **Accuracy is useless here:** Fraud happens in only $0.1727\%$ of transactions. If a dumb model labels literally every transaction as "Legitimate", it gets **$99.8273\%$ accuracy**—while catching $0$ frauds and allowing millions in losses.
- **ROC-AUC is deceptive on imbalanced data:** ROC-AUC plots True Positive Rate (Recall) against False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). The number of legitimate transactions is huge ($284,315$). If our model triggers $1,000$ false alarms, $\text{FPR} = \frac{1000}{1000 + 284315} \approx 0.0035$ (less than $0.4\%$). The ROC curve barely flinches and stays above $0.95$, hiding the fact that investigators are drowning in false alarms.
- **PR-AUC is unsparing:** Precision is $\frac{\text{TP}}{\text{TP} + \text{FP}}$. When false alarms increase, precision drops sharply. The random baseline for PR-AUC is equal to the fraud base rate ($0.0017$). Any model that scores above $0.0017$ is doing real work; achieving $0.60$ or $0.70$ is strong performance. (Reference: Saito & Rehmsmeier 2015; Davis & Goadrich 2006).

### Q2: What does the 0.1727% base rate mean operationally?
- Out of $284,807$ total credit card transactions, exactly $492$ are fraud. That is roughly **1 fraud in every 578 transactions**.
- In practical bank operations, human fraud analysts (the Security Operations Center or SOC) have a fixed queue capacity per day (e.g., they can manually triage $50$, $100$, or $200$ flagged transactions per shift).
- That is why we evaluate **Precision@$k$** and **Recall@$k$**: if the top $100$ alerts contain $80$ actual frauds, the analyst's time is well spent ($80\%$ precision at capacity $100$).

### Q3: Why do PCA features (V1–V28) limit business interpretation?
- The dataset publisher (Worldline & ULB) transformed the original raw banking attributes (e.g., merchant category code, terminal location, cardholder home address, device ID, card velocity) into $28$ orthogonal principal components using PCA.
- **Why was this done?** To protect European cardholder privacy and comply with financial confidentiality regulations (banking secrecy).
- **The limitation:** Each $V_i$ is a linear combination of unknown raw variables. If the model flags a transaction because $V_{14} < -7.0$, we cannot tell the cardholder or merchant *why* in plain language (e.g., "abnormal merchant in another country"). We can only observe mathematical anomaly, not business causality.

### Q4: What temporal drift did we find across the 48-hour timeline?
- The dataset covers exactly $172,792$ seconds ($\approx 48.0$ hours).
- **Circadian pattern:** Transaction volume drops sharply between 2 AM and 6 AM, then peaks in late afternoon. However, fraud attempts happen continuously around the clock, meaning fraud concentration spikes during early morning hours when overall volume drops.
- **Chronological split differences:** 
  - Train (first $33.4$ hours): $170,884$ rows, $360$ frauds ($0.2107\%$ fraud rate).
  - Validation (next $6.9$ hours): $56,961$ rows, $57$ frauds ($0.1001\%$ fraud rate).
  - Test (last $7.7$ hours): $56,962$ rows, $75$ frauds ($0.1317\%$ fraud rate).
- **Two-sample Kolmogorov-Smirnov (KS) tests:** Every single feature exhibits statistically significant distributional differences ($p < 0.05$) between train and test splits. This proves real-world transaction streams are non-stationary, and random k-fold cross-validation would cause severe temporal data leakage.

### Q5: Why does EDA say the GMM assumption is / isn't reasonable?
- A standard Gaussian Mixture Model assumes each cluster is distributed according to a multivariate Gaussian $\mathcal{N}(\mu_k, \Sigma_k)$.
- **What our tests found:**
  1. The Shapiro-Wilk normality test on legitimate training rows rejects univariate normality for **all 28 features** ($p < 0.05$).
  2. Features like $V_2$, $V_7$, and $V_8$ have massive excess kurtosis ($114.9$, $77.0$, and $261.7$), meaning they have super heavy tails, far thicker than a normal distribution bell curve.
  3. Only $9$ out of $28$ features meet mild plausibility criteria ($|\text{skew}| < 1$ and $|\text{excess kurtosis}| < 3$).
- **The verdict:** A single Gaussian ($K=1$) is completely invalid. However, a **mixture** of Gaussians ($K \ge 3$ or $K=5$) with diagonal covariance is a pragmatic approximation because mixtures of Gaussians can approximate arbitrary smooth multi-modal shapes, including heavy tails, by dedicating specific components to tail regions. Furthermore, the correlation between $V$ features is near zero ($|r| \le 0.145$), which validates using diagonal covariance ($\Sigma = \text{diag}(\sigma_1^2, \dots, \sigma_d^2)$).

### Q6: What does each literature reference contribute?
- **Dempster, Laird, Rubin (1977):** The foundational EM algorithm for finding maximum-likelihood parameters in Gaussian Mixture Models with latent variables.
- **Akaike (1974) & Schwarz (1978):** AIC and BIC criteria used for penalizing model complexity to pick the optimal number of mixture components $K$.
- **Liu, Ting, Zhou (2008):** Isolation Forest—our primary tree-based non-parametric anomaly baseline that cuts space randomly to isolate outliers.
- **Schölkopf et al. (2001):** One-Class SVM—our kernel-boundary baseline mapping points into an RKHS to enclose normal transactions.
- **Shyu et al. (2003):** PCA Reconstruction Error baseline—measuring anomaly as residual distance from the normal principal hyperplane.
- **Dal Pozzolo et al. (2015, 2018) & Le Borgne et al. (2022):** The creators of the benchmark dataset, establishing the realities of card fraud: imbalance, concept drift, and verification delay.
- **Saito & Rehmsmeier (2015) & Davis & Goadrich (2006):** Mathematical proofs showing why PR curves are required instead of ROC curves under severe class imbalance.

---

## 2. Walkthrough of Every Generated Plot

### 1. `results/figures/eda_class_imbalance.png`
- **What it shows:** Two subplots. Left: Raw counts of legitimate ($284,315$) vs. fraudulent ($492$) transactions. Right: The exact same counts on a logarithmic scale ($\log_{10}$ y-axis).
- **Why it matters:** On linear axes, the red fraud bar is invisible—it looks like a flat zero line. The log-scale plot is necessary to visually communicate that frauds do exist, but are outnumbered nearly 600 to 1.
- **Viva explanation:** "This demonstrates the severity of the class imbalance ($0.1727\%$). Any metric that weighs majority and minority errors equally, such as classification accuracy, will fail to evaluate fraud triage."

### 2. `results/figures/eda_amount.png`
- **What it shows:** Four-panel layout:
  1. Top-left: Raw histogram of transaction Amount ($0$ to $\$25,000+$). Overwhelmingly clustered below $\$100$.
  2. Top-right: Histogram of $\ln(1 + \text{Amount})$ split by class (blue = legit, red = fraud).
  3. Bottom-left: Side-by-side boxplot of $\ln(1 + \text{Amount})$ for Legit vs Fraud.
  4. Bottom-right: Legit transactions histogram highlighting the $31,862$ IQR outliers above $\$184.15$.
- **Key numbers:**
  - Legit: Mean $\$88.29$, median $\$22.00$, max $\$25,691.16$.
  - Fraud: Mean $\$122.21$, median $\$9.25$, max $\$2,125.87$.
  - Zero-amount transactions: $1,825$ rows ($0.64\%$).
- **Viva explanation:** "Fraudsters exhibit bimodal behavior: either tiny micro-charges (median $\$9.25$) to test stolen cards without triggering SMS alerts, or large cash-out transactions. Log-transforming Amount ($\ln(1 + x)$) is essential to pull in the extreme positive skew."

### 3. `results/figures/eda_time.png`
- **What it shows:** Three-panel plot across the 48-hour timeline:
  1. Top: Total transactions per hour, showing two distinct 24-hour day/night cycles with troughs at night.
  2. Middle: Fraudulent transactions per hour.
  3. Bottom: Fraud percentage per hour (calculated as $\frac{\text{fraud}_h}{\text{total}_h} \times 100$).
- **Key numbers:** Span is $172,792$ seconds ($48.0$ hours). Peak fraud rate occurs between hours 2 AM and 5 AM.
- **Viva explanation:** "While honest transactions drop by $80\%$ during early morning hours, fraudulent transactions do not stop. The relative fraud risk is highest between 2 AM and 6 AM when banking staff are offline. This temporal variation proves that time order cannot be shuffled."

### 4. `results/figures/eda_drift_ks.png`
- **What it shows:** Horizontal bar chart displaying the Kolmogorov-Smirnov (KS) drift statistic for every feature, comparing the training split against the validation split (blue) and test split (orange).
- **Key numbers:** All $30$ features have $p < 0.05$. Amount and Time exhibit substantial distribution differences over the 48-hour period.
- **Viva explanation:** "This plot provides quantitative evidence of concept drift. Even over just 48 hours, feature distributions change significantly. A model trained on Day 1 encounters a shifted distribution on Day 2."

### 5. `results/figures/eda_v_kde_grid.png`
- **What it shows:** A $7 \times 4$ grid of kernel density estimation (KDE) plots for all $28$ PCA features (V1 through V28). Blue curve = legitimate transactions; red dashed curve = fraudulent transactions.
- **Key findings:**
  - Strongly separating features: $V_{14}$, $V_{10}$, $V_{12}$, $V_{17}$, and $V_{11}$. For $V_{14}$, legitimate transactions center around $+0.1$, while frauds cluster in an extreme negative hump around $-7.3$.
  - Uninformative features: $V_{22}$, $V_{23}$, $V_{25}$ have almost identical blue and red curves.
- **Viva explanation:** "This grid shows which dimensions contain anomaly signal. Features like $V_{14}$, $V_{10}$, and $V_{12}$ show massive displacement between legitimate transactions and frauds, with Cohen's $d > 2.0$. These features allow unsupervised density models to identify fraud without ever seeing fraud labels during training."

### 6. `results/figures/eda_corr_heatmap.png`
- **What it shows:** A $28 \times 28$ correlation heatmap showing Pearson correlation coefficients between $V_1$ and $V_{28}$, computed exclusively on legitimate training rows.
- **Key numbers:** Every off-diagonal entry is near zero (maximum $|r| = 0.1451$).
- **Viva explanation:** "The near-zero off-diagonal correlation is expected because $V_1$ through $V_{28}$ are principal components from PCA, which are mathematically orthogonal. This directly justifies using a diagonal covariance matrix ($\text{covariance\_type='diag'}$) in GMM, reducing parameter count from $\mathcal{O}(D^2)$ to $\mathcal{O}(D)$ and preventing covariance matrix singularity."

### 7. `results/figures/eda_pca_2d.png`
- **What it shows:** A 2-D PCA projection of $10,000$ randomly subsampled legitimate transactions (blue dots) alongside all $492$ fraudulent transactions (red points).
- **Key numbers:** PC1 explains $11.6\%$ variance; PC2 explains $9.3\%$.
- **Viva explanation:** "This plot illustrates the triage challenge. Frauds are not neatly isolated in a distant cluster; many lie right inside the dense cloud of normal spending, while others spread into outer tails. A simple linear cutoff cannot catch them; we need flexible density envelopes."

### 8. `results/figures/eda_qq_plots.png`
- **What it shows:** Normal Quantile-Quantile (Q-Q) plots for 4 representative features ($V_4$, $V_{11}$, $V_{14}$, $V_{17}$) on legitimate training instances against theoretical standard normal quantiles.
- **Key findings:**
  - $V_{11}$ adheres closely to the 45-degree red line (near-Gaussian).
  - $V_4$, $V_{14}$, and $V_{17}$ curve away at the extremities, displaying heavy S-curves that indicate fat tails.
- **Viva explanation:** "The Q-Q plots provide visual proof that PCA features have heavier tails than a single Gaussian distribution. This proves that single Gaussian models are insufficient and empirical mixtures ($K \ge 3$) are required."

---

## 3. Walkthrough of Every Generated Table

| CSV Table | Rows $\times$ Cols | What it Contains | Key Takeaway for Viva |
|---|---|---|---|
| `eda_dataset_summary.csv` | $1 \times 8$ | Shape ($284,807 \times 31$), missing count ($0$), duplicate count ($1,081$), fraud count ($492$), memory ($67.36\,\text{MB}$). | "No missing data. Memory footprint is compact ($67\,\text{MB}$). Exactly $1,081$ duplicates." |
| `eda_duplicates.csv` | $1 \times 5$ | Detailed duplicate count ($1,854$ duplicate rows: $1,822$ legit, $32$ fraud, cross-class = True). | "Duplicates were analyzed and kept to mirror real telemetry, exactly as configured." |
| `eda_class_imbalance.csv` | $4 \times 5$ | Breakdown of rows, frauds, and base rate per split (Train: $0.21\%$, Val: $0.10\%$, Test: $0.13\%$, Total: $0.17\%$). | "Base rate drops by half in validation, proving temporal non-stationarity." |
| `eda_amount_stats.csv` | $1 \times 13$ | Mean, median, std, max, IQR bounds, zero-amount count ($1,825$) by class. | "Fraud median is lower ($\$9.25$) than legit ($\$22$), but fraud mean is higher ($\$122$ vs $\$88$)." |
| `eda_time_stats.csv` | $1 \times 5$ | Min Time ($0\,\text{s}$), Max Time ($172,792\,\text{s}$), total span ($48\,\text{h}$). | "Covers exactly 2 full days of European transaction data." |
| `eda_drift_ks.csv` | $30 \times 7$ | KS statistics and p-values comparing train vs val and train vs test for all 30 features. | "All 30 features have statistically significant drift ($p < 0.05$)." |
| `eda_v_feature_ranking.csv` | $28 \times 6$ | Features ranked by KS statistic and Cohen's $d$ between legit and fraud. | "Top discriminators are $V_{14}$ ($\text{KS}=0.85, d=2.32$), $V_{10}$ ($\text{KS}=0.83$), $V_{12}$ ($\text{KS}=0.80$)." |
| `eda_corr_matrix.csv` | $28 \times 28$ | Pairwise Pearson correlations for V1--V28 on legit training transactions. | "Confirms orthogonality of PCA features (maximum $|r| = 0.145$)." |
| `eda_gaussianity.csv` | $28 \times 10$ | Mean, std, skewness, excess kurtosis, Shapiro-Wilk stat & p-value for each V feature. | "All 28 reject univariate normality. 9/28 are moderately plausible; others have heavy kurtosis." |
| `eda_leakage_audit.csv` | $7 \times 3$ | Audit of 7 strict rules preventing data leakage across time and class boundaries. | "All 7 checks passed. Scaler fit only on train-legit; zero leakage." |

---

## 4. How We Prevented Data Leakage (Audit Checklist)

If the examiner asks: *"How do you know your model didn't cheat or overfit?"*

1. **Chronological Splitting:** The data is split on the `Time` column:
   - Train: first $60\%$ ($[0, 120\,396\,\text{s}]$)
   - Val: next $20\%$ ($[120\,396, 145\,247\,\text{s}]$)
   - Test: final $20\%$ ($[145\,248, 172\,792\,\text{s}]$)
   No random shuffling was used. The test set is from the future relative to training data.
2. **StandardScaler isolation:** The scaler is fit strictly on `X_train_legit` (only legitimate rows in the training set). It never sees validation data, test data, or fraudulent transactions during fitting.
3. **Unsupervised Fitting:** The anomaly detection models (GMM, Isolation Forest, OC-SVM, PCA) are trained without seeing fraud labels.
4. **Validation-selected Thresholds:** Decision thresholds are selected on the validation split (e.g., targeting operational capacity $k=100$ or best validation F1) and then evaluated on the frozen test set without adjustment.
5. **Raw Data for EDA:** All EDA calculations are computed on raw feature values, not on scaled data.
