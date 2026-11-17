# Card Fraud Triage

Team T033 · Section A · Rishihood University · Project 26

The project ranks transactions for fraud review on the ULB Credit Card Fraud Detection dataset. Phase 1 compares ambient Gaussian mixture density with classical detectors. Phase 2 adds an undercomplete autoencoder, a denoising autoencoder (DAE), and three hybrids. The experiments use a chronological 60/20/20 split. Model fitting uses 170,524 legitimate training transactions. The validation and test partitions contain 57 and 75 frauds, respectively.

## Run the Phase 2 demo

Run from the repository root. Python 3.12 was used for the audit.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase2.txt
mkdir -p data/raw
```

Download `creditcard.csv` from the [ULB dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), unzip it, and place it at `data/raw/creditcard.csv`. Keep the default split configuration with duplicate handling unchanged for the supplied scores.

```bash
python -m src.final_phase2
python -m pytest -q
python -m jupyter lab
```

Open [06b_final_comparison_phase2.ipynb](notebooks/06b_final_comparison_phase2.ipynb) in Jupyter. The evaluation command rebuilds missing splits, reads the stored anomaly scores, and writes the comparison, threshold table, and plots. The tests use the generated splits. The notebook displays the comparison and runs paired bootstrap estimates.

## Rebuild experiments from the dataset

The following stages can overwrite saved scores and figures. Use a second checkout if you want to retain the submitted snapshots. Some sweeps are computationally intensive. Results can vary by library version and device.

```bash
python run_all.py
python -m src.eda_dl
python -m src.train_ae
python -m src.hybrid_latent_gmm
python -m src.hybrid_fusion
python -m src.hybrid_dagmm_lite
python -m src.ablation
python -m src.final_phase2
```

`run_all.py` runs the classical stages. The AE stage produces the checkpoints used by H1 and H3. H2 reads DAE and GMM scores. H3 also compares the other hybrid scores and should run after H1 and H2. The rebuild commands are provided for experimentation. The submission audit verified the saved-score workflow.

## Results

| Model | Test AP | Precision at 100 | Fraud caught at 0.5% validation FPR target | False alarms |
|---|---:|---:|---:|---:|
| PCA reconstruction | 0.7425 | 57% | 58/75 | 332 |
| Stored ambient GMM | 0.2679 | 31% | 46/75 | 196 |
| Standard AE | 0.3574 | 40% | 50/75 | 259 |
| Denoising AE / H2 | 0.4818 | 53% | 58/75 | 186 |
| H1 latent GMM | 0.1724 | 30% | 53/75 | 206 |
| H3 DAGMM-lite | 0.4541 | 48% | 59/75 | 197 |

AP is average precision, labeled PR-AUC in the code. At the 0.5% validation target, realized test FPR was 0.346% for H3 and 0.327% for DAE. H3 catches 60/75 at the separate 0.5% point of the test ROC curve. H2 preserves the DAE ranking because validation selected zero weight for GMM. The supplied Phase 1 report lists GMM AP as 0.4142. The Phase 2 comparison and paired intervals use the repository score arrays, which give 0.2679.

- [Ten-model comparison](results/tables/final_comparison_phase2.csv)
- [Validation thresholds and test workload](results/tables/final_thresholds_phase2.csv)
- [Test precision–recall curves](results/figures/final_phase2_pr_curves.png)
- [Recall under fixed validation thresholds](results/figures/final_phase2_recall_at_fpr.png)

## Team members and contributions

| Team member | Enrollment no. | Phase 1 · Classical machine learning | Phase 2 · Deep learning and hybrid models |
|---|---|---|---|
| Kushal Sarkar | 2401020033 | Set up the project and implemented data loading, chronological splits, and shared metrics. Integrated the Phase 1 results and report. | Integrated the model pipelines, ran the ten-model comparison and paired bootstrap tests, and coordinated the final report. |
| Prachee Dhar | 2401010330 | Reviewed related work and explored the data, checking duplicates, class imbalance, and changes over time. | Built H2 score fusion and H3 DAGMM-lite. Compared alert overlap and tested how model settings affected results. |
| Apoorva Choudhary | 2401010092 | Designed input features and tested GMM settings using AIC, BIC, and validation results. Studied when fitting became unstable. | Built H1, the Gaussian mixture on autoencoder codes. Tested covariance choices and studied why some frauds received low scores. |
| Vani Rudra | 2401010490 | Built and evaluated the PCA, Local Outlier Factor, One-Class SVM, and Isolation Forest baselines. | Trained the standard and denoising autoencoders on genuine payments. Compared five random starts and profiled scoring time. |
| Krishna Pramod Dubale | 2401010239 | Built the evaluation toolkit for alert thresholds, confidence intervals, and review costs. Assembled the comparison table. | Reviewed deep-learning research and checked time-based splits, feature scaling, and how genuine validation rows were used to stop training. |

## Submission materials

- [Phase 2 IEEE report PDF](reports/T033_Phase2_IEEE_Report.pdf) and [Phase 2 IEEE LaTeX source](reports/ieee_report/main_phase2.tex) and [bibliography](reports/ieee_report/refs_phase2.bib)
- [Phase 2 presentation](docs/T033_Phase2_Presentation.pdf)
- [Contributions](CONTRIBUTIONS.md)
- [Phase 1 report](reports/T033_Phase1_IEEE_Report.pdf) and [slides](docs/T033_Phase1_Presentation_Slides.pdf)
