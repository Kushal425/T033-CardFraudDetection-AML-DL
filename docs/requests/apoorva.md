# Shared-File & Cross-Team Coordination Requests — Apoorva Choudhary (`P2_02`)

**Branch:** `feat/apoorva-p2`
**Scope:** Latent-Space Gaussian Mixture Model (Hybrid H1, Phase 2)

This file tracks all cross-owner interface agreements and requested edits to shared repository files that `P2_02` does not own directly.

---

## 1. Requests to Kushal (`P2_05` — Final Integration & Report Assembly)

1. **Include Apoorva's Phase 2 IEEE LaTeX sections in `reports/ieee_report/main_phase2.tex`:**
   - `\input{sections/p2_hybrid_theory}` (`\label{sec:p2_hybrid_theory}`) — mathematical formulation of $x \to z$, GMM likelihood, EM responsibilities, covariance structures, `reg_covar` eigenvalue bounding, NLL anomaly score, and density--manifold pathologies.
   - `\input{sections/p2_latent_gmm_results}` (`\label{sec:p2_latent_gmm_results}`) — experimental configuration selection (BIC vs Validation PR-AUC), parent comparison, conditioning/seed stability, latent non-Gaussianity diagnostics, complementarity, and bottleneck-dimension ablation ($k \in \{2,4,8,12,16\}$).

2. **Merge BibTeX references into the shared Phase 2 bibliography (`reports/ieee_report/refs.bib` / `reports/refs.bib`):**
   - Source file: `reports/refs_apoorva_p2.bib`
   - Keys cited in Apoorva's sections: `dempster1977maximum`, `akaike1974new`, `schwarz1978estimating`, `bishop2006pattern`, `mclachlan2000finite`, `aggarwal2017outlier`, `hinton2006reducing`, `sakurada2014anomaly`, `zong2018deep`, `gong2019memorizing`, `rezende2015variational`, `chandola2009anomaly`.
   - Note: Please verify bibliographic venue details for `rezende2015variational` (marked with `[VERIFY]` note) during final Phase 2 report bibliography audit.

3. **Register `hybrid_latentgmm` in shared score contracts / evaluation harness:**
   - Add `"hybrid_latentgmm"` to `REAL_MODELS` in `tests/test_score_contract.py` (not edited by Apoorva because it is a shared file).
   - Register `hybrid_latentgmm` in `src/final.py` / `configs/config.yaml` Phase 2 evaluation tables without overwriting Phase 1 frozen `gmm_{val,test}.npy` arrays.

4. **Pre-existing dirty file in workspace (do not attribute to `feat/apoorva-p2`):**
   - `reports/ieee_report/sections/p1_gmm_results.tex` contains a pre-existing leading `git ` typo on line 1 (`git \section{...}`) from prior to `feat/apoorva-p2` creation. It was intentionally left unstaged and untouched in Apoorva's commits.

---

## 2. Agreed Interface with Vani (`P2_03` — Autoencoder)

- **Encoder module & functions consumed (read-only):** `load_autoencoder(checkpoint_path, config_path)` and `encode(model, X)` from `src/autoencoder.py`.
- **Feature matrix contract:** Standardized $d=31$ `base` features (`get_features("base")`), matching Vani's trained checkpoints.
- **Checkpoints consumed:**
  - Primary H1 model: `results/models/ae_best.pt` + `results/models/ae_config.json` (`UndercompleteAE`, $d_z = 12$, hidden `[32, 16]`, LeakyReLU bottleneck).
  - Multi-dimension ablation checkpoints: `results/models/ae_latent_k{2,4,8,16}.pt` and corresponding configs.
- Zero modifications were made to `src/autoencoder.py`, `src/train_ae.py`, or any `ae_*` artifacts.

---

## 3. Agreed Score Contract & Diagnostic Handoff to Prachee (`P2_04` — Hybrid & Ablation)

- **Official frozen score arrays:**
  - `results/scores/hybrid_latentgmm_val.npy`: shape `(56961,)`, dtype `float32`, finite, higher = more anomalous (`-\ln p(z)` under refitted `diag`, $K=12$, $\varepsilon=10^{-2}$ GMM on all $170{,}524$ legitimate training transactions).
  - `results/scores/hybrid_latentgmm_test.npy`: shape `(56962,)`, dtype `float32`, finite, same index order as `data/processed` val/test splits.
- **Key diagnostic takeaway for H2/H3 (AE reconstruction + Latent GMM fusion):**
  - Standalone H1 on `ae_best.pt` ($d_z=12$) has lower PR-AUC (`0.1724` test) due to LeakyReLU right-skewed latent coordinates (`0.0%` mildly Gaussian), which push extreme benign outliers into the top-50 alert queue (`Precision@50 = 0.02` test).
  - However, at `FPR <= 0.5%`, H1 achieves `0.7467` test recall and catches `8` test frauds missed by `ae_best` reconstruction error (union recall `0.7867`), motivating weighted fusion of normalized reconstruction error and normalized latent NLL (and testing `k=16` `tied` covariance, which reached `0.4149 ± 0.0356` test PR-AUC in the latent ablation).

