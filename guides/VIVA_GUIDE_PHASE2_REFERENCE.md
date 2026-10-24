# VIVA GUIDE — Individual Viva Voce (Team T033, Project 26)

## 1. Format expectations (from instructions)
Individual assessment even though project is team-based. You must explain: **methods, implementation, experiments, results, relevant theory**, and show your own contribution (repo, report, presentation). Expect: "open your code and walk me through", "change this parameter and predict what happens", "why not X?", "what does this plot show?".

## 2. Preparation rules
1. Everyone must know the **whole pipeline in 2 minutes** + **own part in depth** (see table).
2. Re-run your own notebook once from scratch before viva. Know runtime, shapes, and file names.
3. Read `docs/viva_notes/<yourname>.md` and rewrite 5 key points in your own words.
4. Know your numbers: base rate, split sizes, best K & cov type, AE latent dim, PR-AUC of GMM/AE/hybrid with CIs.
5. Be honest: "I don't know, but I'd check X" beats bluffing. Mention limitations before they ask.
6. Do 2 mock vivas: members question each other on *each other's* parts (random swap).

| Member | Deep area | Must also know |
|---|---|---|
| Kushal | splits, leakage control, metrics, integration, final comparison, threats to validity | everything at overview level |
| Prachee | literature, EDA, drift, Gaussianity, data limits | why these findings affect GMM/AE |
| Apoorva | FE, EM, AIC/BIC, GMM results & instability | how AE differs |
| Vani | AE architecture, regularization, training, DL lit | PCA link, how GMM scores |
| Krishna | baselines, bootstrap CI, thresholds, hybrids, ablation | both parent models |

## 3. 30-second pitch (memorise)
"We treat card fraud as anomaly detection: models learn only legitimate behaviour from the first 60% of a 48-hour log; validation (next 20%) selects hyper-parameters and thresholds; the last 20% is untouched for testing. A Gaussian Mixture scores unusualness by negative log-likelihood; an undercomplete/denoising autoencoder scores it by reconstruction error. With only ~0.17% fraud we report PR-AUC, recall at fixed false-positive budgets and precision at review capacity, never accuracy. We also built hybrids — a GMM on the autoencoder latent space and score fusion — and ablations to see what actually matters."

## 4. Question bank (with short model answers)
### Whole-project
1. **Why anomaly detection, not a classifier?** Labels are scarce/delayed, 0.17% fraud, fraud patterns change; unsupervised needs only legit data. (A supervised model would be a useful extra baseline but not the task.)
2. **Why not accuracy?** Predicting "all legit" gives 99.83%. Use PR-AUC, recall@FPR, precision@capacity.
3. **Why chronological split?** Mimics deployment; random split leaks future patterns & drift.
4. **Where could leakage occur and how did you prevent it?** Scaler/GMM/AE fit on train-legit only; thresholds on val; test used once; labels not used in fitting.
5. **How were thresholds set?** On val: FPR budget quantile of legit scores / top-K review capacity / max-F1; frozen then applied to test.
6. **Biggest limitation?** Few test frauds → wide CIs; PCA-anonymised features; 2-day window; single dataset.
7. **Is AE significantly better than GMM?** Answer from YOUR paired-bootstrap result; if not significant, say so.
8. **What is the base rate and what does a random model score in PR-AUC?** ≈0.0017 (per-split rate from manifest).

### GMM / theory
9. **Write the GMM density.** p(x)=Σπ_k N(x|μ_k,Σ_k).
10. **E-step / M-step?** Responsibilities γ; then update N_k, μ_k, Σ_k, π_k.
11. **Why does EM increase likelihood?** Maximises a lower bound (ELBO via Jensen); each step doesn't decrease it; local optimum only → n_init.
12. **AIC vs BIC?** AIC=2p−2lnL; BIC=p ln n−2lnL; BIC penalises complexity more for large n.
13. **Parameters of full-cov GMM?** K·d + K·d(d+1)/2 + K−1.
14. **What is covariance singularity?** Component collapses on few points; fix with reg_covar.
15. **Why can GMM fail here?** Non-Gaussian/heavy tails (Amount), d=30 many parameters, density≠fraudness (fraud can sit in dense regions).
16. **Which cov-type & K won and why?** Your table; justify with BIC and val PR-AUC.

### Autoencoder / DL
17. **Why undercomplete?** Bottleneck prevents identity mapping.
18. **Why train only on legit?** The model should reconstruct normal well, anomalies badly.
19. **What does denoising add?** Learns manifold structure, robustness; regularizer.
20. **Linear AE vs PCA?** Same subspace under MSE; nonlinearity adds capacity.
21. **When does AE reconstruction error fail?** Anomalies lying in the principal subspace; AE generalises to anomalies; dominated by large-variance features.
22. **Early stopping — does it leak?** Monitors legit-only val loss, no fraud labels.
23. **Which regularizers did you test, what happened?** From Vani's table.
24. **Why MSE? alternatives?** MAE/Huber, per-feature weighting, Mahalanobis in latent.
25. **Param count / layer shapes?** From model summary.
26. **Why no SMOTE?** Unsupervised legit-only training; synthetic fraud irrelevant & leaks label usage.

### Hybrid / ablation / evaluation
27. **Why should GMM-on-latent help?** Lower dimension stabilises density estimation; AE supplies nonlinear features.
28. **Score fusion — how to normalise and pick w?** Train-legit z/rank normalisation; w on val.
29. **Did hybrid win?** Report honestly with CI.
30. **Most influential ablation factor?** From your table.
31. **Bootstrap CI — how?** Resample test rows with replacement (stratified), recompute metric, take 2.5/97.5 percentiles.
32. **Recall@FPR=0.5% meaning?** Fraction of frauds caught while falsely flagging ≤0.5% of legit.
33. **Precision at review capacity?** Of the top-K flagged (what analysts can review), share that is fraud.
34. **Why PR curve over ROC here?** ROC hides false-positive burden under extreme imbalance.
35. **Cost curve assumptions?** Missed-fraud cost ratio is assumed, swept.

### Live-demo style
36. "Change K to 2 / cov-type to diag — what happens to BIC and PR-AUC?" (Predict, then run.)
37. "Increase latent dim to 28 — what do you expect?" (AE approaches identity → worse separation.)
38. "Show me where you prevent leakage in code."
39. "Show your commits."

## 5. Presentation tips
Everyone speaks ~3 min; handover sentences; one demo (load saved scores → PR curves → threshold table); backup slides ready; never read slides; end with limitations + future work (drift handling, supervised hybrid, IEEE-CIS dataset, cost-sensitive thresholds).
