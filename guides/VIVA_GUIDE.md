# VIVA GUIDE — Phase 1 Only (Team T033, Project 26)

**Current submission scope: Phase 1 Machine Learning evaluation only.** Use this guide for the current evaluation. Deep Learning and hybrid questions are intentionally excluded; the separate `VIVA_GUIDE_PHASE2_REFERENCE.md` is retained for later use.

## 1. What the viva assesses
The official instructions say the viva is individual. Each member must explain the methods, implementation, experiments, results, relevant theory and their own contribution.

## 2. Preparation rules
1. Know the full Phase 1 pipeline in two minutes and your own component in depth.
2. Re-run your notebook from a clean environment and know its inputs, outputs, runtime and limitations.
3. Read `docs/viva_notes/<yourname>.md` and explain the main decisions in your own words.
4. Know the actual measured base rate, split sizes and fraud counts, chosen GMM component count/covariance type, baseline scores and confidence intervals. Never memorize placeholder or unverified values.
5. Be honest about uncertainty and limitations.
6. Conduct mock vivas in which members ask each other questions about the whole Phase 1 pipeline.

| Member | Deep area | Must also know |
|---|---|---|
| Kushal | splits, leakage control, metrics, integration, final comparison, validity threats | overall pipeline and all major results |
| Prachee | literature, EDA, drift, Gaussianity checks, data limitations | how EDA affects model choices |
| Apoorva | feature engineering, EM, AIC/BIC, GMM results and instability | why GMM assumptions may fail |
| Vani | PCA reconstruction, Isolation Forest, One-Class SVM, LOF and random/Amount references | how each baseline generates anomaly scores |
| Krishna | bootstrap confidence intervals, thresholds, cost assumptions and evaluation | how to interpret model comparisons |

## 3. Two-minute project explanation template

“We treat credit-card fraud as an anomaly-detection problem because fraudulent transactions are rare. Models are fitted on legitimate training transactions only. We use a chronological train/validation/test split; the validation set is used for model selection and threshold choices, and the test set is held back for final evaluation. Our main model is a Gaussian Mixture Model, which assigns higher anomaly scores to transactions with lower estimated density. We compare it with classical anomaly-detection baselines using precision-recall metrics, recall at specified false-positive rates, precision at review capacity and uncertainty estimates. We discuss data limitations, possible temporal drift and the operational trade-offs of false alarms versus missed fraud.”

Adapt this template to match the implementation and actual results. Do not claim a model or experiment was run unless it was.

## 4. Core Phase 1 viva questions
1. Why is fraud detection an anomaly-detection problem? What are its limitations?
2. Why is accuracy misleading with an approximately 0.17% fraud base rate?
3. What does PR-AUC / average precision measure, and why report it?
4. What is a Gaussian Mixture Model? Explain its parameters.
5. Explain the EM algorithm: E-step, M-step and convergence.
6. What do AIC and BIC measure? How do their penalties differ?
7. How many parameters does a GMM have for a given number of components and covariance type?
8. Why can GMM covariance estimates become singular or ill-conditioned? What does `reg_covar` do?
9. Why fit preprocessing and anomaly models only on legitimate training rows?
10. Why use a chronological split, and what are the risks with a short two-day dataset?
11. How do you select a threshold using validation data and apply it unchanged to test?
12. Explain recall at a fixed false-positive rate in plain language.
13. What is precision at review capacity? Why might it matter operationally?
14. How does PCA reconstruction error become an anomaly score?
15. How does Isolation Forest identify anomalies using path lengths?
16. What is the One-Class SVM decision boundary, and why is it expensive on large datasets?
17. What does Local Outlier Factor measure, and why is `novelty=True` relevant for unseen data?
18. Why should a random-score baseline have PR-AUC close to the prevalence?
19. What does a bootstrap confidence interval tell us? Why can it be wide?
20. How do you interpret paired-bootstrap differences between two models?
21. What assumptions are made in the cost curve?
22. What are the dataset’s main limitations, including anonymized PCA features and limited time coverage?
23. Show where leakage is prevented in code.
24. Explain your contribution and demonstrate a real commit or pull request.

## 5. Presentation and live demo
Everyone should speak to their own work and be able to explain the overall method. Use actual figures and tables generated from code. Prepare a short demo that loads saved scores, displays the precision-recall curves and threshold table, and explains at least one failure case. Keep limitations and individual contributions visible.
