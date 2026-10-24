# PHASE 1 ONLY — Team T033, Project 26

**Current submission: Phase 1 Machine Learning evaluation only.** The official evaluation criteria are literature review, dataset quality and EDA, feature engineering, theoretical rigor, model application, GitHub repository/code quality, presentation, individual viva and an IEEE-format project report.

## Scope for this submission
- Implement and evaluate GMM plus classical ML anomaly-detection baselines (PCA reconstruction error, Isolation Forest, One-Class SVM, LOF, and clearly labelled simple reference scores).
- Do not implement an autoencoder, neural network, hybrid model or DL ablation for this submission. The `prompts_phase2/` folder is retained for later and must not be treated as a Phase 1 requirement.
- A single brief future-work sentence may say that a later phase could compare neural reconstruction scores; no Phase 2 implementation or results belong in the current report.
- Train anomaly models on legitimate training transactions only. Use validation data for model selection and threshold selection, and evaluate on the untouched mixed test set. Never fit on fraud labels.

## Phase 1 roles
| Member | Phase 1 task | Current prompt file |
|---|---|---|
| Kushal Sarkar | Scaffold, splits, metrics, integration, discussion, report assembly | `prompts_phase1/P1_0A_Kushal_START.md`, then `prompts_phase1/P1_05_Kushal_FINAL.md` |
| Prachee Dhar | Literature review, EDA, dataset quality | `prompts_phase1/P1_01_Prachee_Literature_EDA.md` |
| Apoorva Choudhary | Feature engineering, GMM grid, AIC/BIC, EM theory and instability study | `prompts_phase1/P1_02_Apoorva_FeatureEng_GMM.md` |
| Vani Rudra | Classical anomaly-detection baselines and their theory | `prompts_phase1/P1_03_Vani_Classical_Baselines.md` |
| Krishna Pramod Dubale | Evaluation toolkit, bootstrap CIs, threshold policies and comparisons | `prompts_phase1/P1_04_Krishna_Evaluation_Toolkit.md` |

## Execution order
1. Kushal builds the shared scaffold and merges it to `main`; teammates then pull it.
2. Prachee, Apoorva, Vani and Krishna work on their assigned files in parallel, using separate branches and the shared score/metric contracts.
3. Integrate the GMM and baseline scores, then run the evaluation toolkit and final comparison.
4. Verify the report numbers against generated CSVs, compile the Phase 1 IEEE report, prepare slides, run tests and conduct a mock viva.
5. Tag `phase1-final` only when the actual Phase 1 artifacts have been reviewed and validated.

## Important technical checks
- **Split counts:** The proposed 60/20/20 chronological split is a starting point, not a guaranteed statistically adequate split. Inspect `split_manifest.json`, including fraud counts per split. If validation or test has too few frauds for stable estimates, discuss and document a split change before model selection; never silently change the split after seeing test performance.
- **Thresholds:** A threshold described as an FPR budget must be estimated from validation legitimate scores, then frozen and applied to test. Report the achieved test FPR rather than assuming it will exactly equal the target.
- **Metrics:** PR-AUC/average precision, recall at FPR budgets, precision at review capacity, confidence intervals and false-positive analysis should lead the results. Accuracy must not be the headline.
- **Data:** Do not commit the raw dataset or large model files. Document how to obtain the dataset and reproduce results.
- **Commits:** Commit real, meaningful work as it is completed. Do not fabricate commits, manipulate timestamps or rewrite history. Multiple commits are useful, but do not require a fixed number of calendar days if the actual schedule does not permit it.

## Phase 1 report outline
Abstract; Introduction; Related Work; Dataset & EDA; Feature Engineering; GMM Theory (EM, AIC/BIC); Classical Baselines; Experimental Protocol; Results (PR-AUC, recall@FPR, precision@capacity, threshold analysis and confidence intervals); Discussion & Limitations; Conclusion; Contributions; References.

The ZIP proposes 6–8 pages and 12–15 slides as working targets. These are not explicit limits in the official instructions; confirm any required format, page/slide limits, portal and deadline with faculty.

## Phase 1 submission preparation checklist
- [ ] GitHub repository link and reviewed `phase1-final` tag
- [ ] Phase 1 report PDF and its LaTeX source (IEEE format)
- [ ] Phase 1 presentation
- [ ] Executed relevant notebooks with real outputs
- [ ] README with setup and reproduction instructions; tests pass or documented limitations
- [ ] Individual viva preparation and genuine contribution evidence for all five members
- [ ] Submission channel and deadline confirmed with faculty
