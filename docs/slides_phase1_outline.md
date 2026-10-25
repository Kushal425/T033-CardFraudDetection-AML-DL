# Phase 1 Slides Outline — Team T033, Project 26
**Card Fraud Triage: GMM Density Scoring vs Classical Baselines**
12–15 slides · ~3 min per member · Each slide = one message

---

## Who speaks what

| Slides | Member | Topic |
|---|---|---|
| 1–3 | Kushal | Problem, dataset, project structure |
| 4–5 | Prachee | Literature, EDA findings |
| 6–7 | Apoorva | Feature engineering, GMM theory & results |
| 8–9 | Vani | Classical baselines theory & results |
| 10–11 | Krishna | Evaluation protocol, results, threshold analysis |
| 12–13 | Kushal | Final comparison, discussion, limitations |
| 14–15 | All | Contributions, Q&A backup |

---

## Slide-by-Slide Outline

### Slide 1 — Problem Statement *(Kushal)*
**Message:** Fraud is 0.17% of transactions — standard classification fails.
- Title: "Card Fraud Triage: Finding 1 in 600 Transactions"
- Problem: Class imbalance, PCA-anonymised features, real-time deployment
- Our approach: anomaly detection — train on legitimate only, score by deviation
- Key question: which anomaly scoring method best catches fraud at low false-alarm rate?
- Visual: donut chart of 99.83% legit vs 0.17% fraud

### Slide 2 — Dataset & Splits *(Kushal)*
**Message:** ULB Kaggle dataset — 284,807 transactions, 48 hours, chronological split.
- Dataset: Time, V1–V28 (PCA), Amount, Class
- Split: 60/20/20 chronological (NO shuffle — prevents future leakage)
- Table: train/val/test rows, frauds, base rates
- Show split_manifest.json excerpt
- Highlight: "test set is sacred — never touched until final evaluation"

### Slide 3 — Project Architecture *(Kushal)*
**Message:** Modular pipeline with strict leakage rules.
- Show repo structure diagram
- Shared contracts: score files (higher = more anomalous), metrics module
- Key rule: scaler/models fit on train-legit only; fraud labels hidden
- Run: `python run_all.py`
- Highlight: 80 automated tests

### Slide 4 — Literature Review *(Prachee)*
**Message:** Anomaly detection is mature; GMM and autoencoders are the natural pair.
- Table of ≥5 key references with gap column
- Dal Pozzolo et al. 2015 (dataset), Chandola et al. 2009 (survey)
- Gap: GMM assumes Gaussian manifold, high-dim instability
- Why not supervised? Labels expensive + imbalance inflates accuracy
- Transition: "So we frame it as density estimation"

### Slide 5 — EDA Findings *(Prachee)*
**Message:** Data is imbalanced, time-drifted, and Amount is heavy-tailed.
- Show: class imbalance bar + Amount distribution
- KS test: train vs test distribution drift for top V-features
- Duplicates: 1,081 rows (0.38%) — kept, logged
- Gaussianity: most V-features approximately Gaussian (table)
- Key limitation: V1–V28 are already PCA → feature engineering is limited

### Slide 6 — Feature Engineering *(Apoorva)*
**Message:** Base features work best; PCA already done.
- Show feature set comparison table (raw vs base vs engineered) with val PR-AUC
- Base: V1–V28 + log1p(Amount) + sin/cos(time) — 31 features
- Why log1p(Amount)? Compresses heavy tail
- Why sin/cos time? Cyclic encoding avoids jump at midnight
- Winner: base feature set; V-features are already decorrelated

### Slide 7 — GMM Theory & Results *(Apoorva)*
**Message:** GMM fits the legitimate distribution; low probability = anomaly.
- EM algorithm: E-step (responsibilities) + M-step (update μ, Σ, π)
- AIC/BIC plot: show optimal K selection (use actual figure)
- Instability study: variance of PR-AUC across 5 seeds
- Score = −log p(x): show NLL histogram legit vs fraud
- Result: PR-AUC (test) = **0.4142** — above random but below PCA

### Slide 8 — Classical Baselines Theory *(Vani)*
**Message:** Four classical methods, each with a different anomaly intuition.
| Model | Anomaly intuition | Key hyper-param |
|---|---|---|
| PCA Recon. | High reconstruction error = anomalous | n_components |
| Isolation Forest | Hard to isolate = normal; easy = anomalous | n_estimators, contamination |
| One-Class SVM | Outside learned boundary = anomalous | ν, γ |
| LOF | Low local density ratio = anomalous | n_neighbors |
- All fit on train-legit only; score convention: higher = more anomalous

### Slide 9 — Baselines Results *(Vani)*
**Message:** PCA reconstruction is the clear winner.
- Show bar chart of PR-AUC per model (val and test)
- Timing table: OC-SVM is slowest (58s), Iso. Forest 14s
- Rank correlation heatmap of scores — GMM and LOF most correlated
- Surprise: Isolation Forest underperforms even Amount-only baseline on some metrics

### Slide 10 — Evaluation Protocol *(Krishna)*
**Message:** PR-AUC, not accuracy — with confidence intervals and threshold policies.
- Why PR-AUC? ROC-AUC is inflated by true negatives under imbalance
- Bootstrap CI procedure (1000 resamples, stratified)
- Three threshold policies: FPR budget / review capacity / max-F1
- Cost curve: sweep missed-fraud cost ratio 10–500×
- "We never peeked at test to tune any threshold"

### Slide 11 — Threshold Analysis *(Krishna)*
**Message:** At FPR=0.5%, PCA catches 77% of fraud; GMM catches 69%.
- Show threshold table (FPR budget policy, val → test)
- Show precision@top-K table (top 100, 200 transactions)
- Temporal stability: val vs test PR-AUC per model
- False-alert burden: ~285 false alarms per 56,962 transactions for PCA@FPR=0.5%
- Show: `eval_threshold_fpr.csv` screenshot

### Slide 12 — Final Comparison *(Kushal)*
**Message:** PCA wins. GMM has high gap between val and test — instability confirmed.
- Show headline table with CIs
- PR-curve plot (all models, same axes) — from `final_pr_curves.png`
- Recall@FPR bar chart — `final_recall_at_fpr.png`
- Statistical significance: only PCA is sig. better (paired bootstrap vs rest)
- Key finding: GMM val/test PR-AUC gap (0.195 → 0.414) = instability evidence

### Slide 13 — Discussion & Limitations *(Kushal)*
**Message:** Good results but important caveats — don't over-claim.
- Threats to validity: 75 test frauds → wide CIs
- PCA-anonymised features limit interpretability
- 48-hour window: no seasonal patterns
- Assumed cost ratios in cost-curve analysis
- **Disclaimer**: research prototype — not for production decisions
- Future work (one sentence): Phase 2 will compare a deep autoencoder

### Slide 14 — Individual Contributions *(All)*
**Message:** Every member built a distinct, meaningful component.
- Wheel/table showing who owns what
- Screenshot of `git log --oneline` showing commits from all 5 members
- Mention: 80 automated tests, no fabricated numbers, leakage rules enforced
- "Each of us can explain our module live"

### Slide 15 — Q&A + Backup *(All)*
**Message:** We are prepared.
**Backup slides (have ready):**
- EM derivation (whiteboard-style)
- AIC/BIC formula + K selection explanation
- Score convention proof (why higher = more anomalous)
- Threshold derivation from val-legit quantile
- Bootstrap CI steps
- Data card (duplicate policy, split policy, known limitations)

---

## Presentation Tips

1. **Demo (Slide 3):** Open terminal live and run `python run_all.py` — it prints OK for scaffold and shows shapes.
2. **Figures are already generated** in `results/figures/` — embed directly.
3. **Every number you say must come from a CSV in results/tables/** — the faculty may ask "where does that number come from?"
4. **Viva prep:** each person must be able to explain their module without notes.
