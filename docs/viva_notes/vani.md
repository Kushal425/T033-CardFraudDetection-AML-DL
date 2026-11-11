# Viva Defense Notes — Vani Rudra (Phase 2: Deep Autoencoder)

**Project:** 26 — Card Fraud Triage (GMM vs Autoencoder) | **Team:** T033  
**Task:** `P2_03_Vani_Autoencoder.md` | **Branch:** `feat/vani-p2`

---

## 1. Architecture & Hyperparameters

### Final Selected Models (Base Feature Set, `d_in = 31`)
- **Undercomplete Autoencoder (`UndercompleteAE`):**
  - **Shape:** `31 → 32 → 16 → 12 → 16 → 32 → 31` (`latent_dim = 12`, `hidden = [32, 16]`)
  - **Activations:** `LeakyReLU` on all hidden layers + bottleneck; **linear** output layer (unbounded real-valued standardized targets)
  - **Initialization:** Kaiming/He normal initialization for ReLU-family layers, zero bias
  - **Regularization & Optimizer:** `dropout = 0.0`, `weight_decay = 1e-5`, Adam (`lr = 1e-3`), `batch_size = 512`, gradient clipping `max_norm = 5.0`, `ReduceLROnPlateau(factor=0.5, patience=5)`, early stopping `patience = 10` on **val-legit MSE**
  - **Trainable Parameters:** `3,531` (`(31×32+32) + (32×16+16) + (16×12+12) + (12×16+16) + (16×32+32) + (32×31+31) = 1024 + 528 + 204 + 208 + 544 + 1023 = 3531`)
  - **Official Median Seed (by val PR-AUC):** `1024`
- **Denoising Autoencoder (`DenoisingAE`):**
  - **Shape:** `31 → 24 → 16 → 16 → 16 → 24 → 31` (`latent_dim = 16`, `hidden = [24, 16]`)
  - **Activations:** `LeakyReLU`, linear output
  - **Noise Corruption:** `x̃ = x + σ·ε` with `σ = 0.2`, `ε ~ N(0, I)` during training only; loss compares reconstruction to **clean** `x`; **zero noise at inference**
  - **Trainable Parameters:** `2,895`
  - **Official Median Seed (by val PR-AUC):** `123`

---

## 2. Plain-English Walkthrough of Every Function & Class

### `src/autoencoder.py`
- **`_build_layers(dims, act, dropout, batchnorm, final_activation)`**: Helper that constructs a sequential MLP block (`Linear → [BatchNorm1d] → Activation → [Dropout]`) given layer sizes. Leaves out activation/dropout/batchnorm on the final decoder layer when `final_activation=False`.
- **`UndercompleteAE`**: Symmetric PyTorch `nn.Module` with `encoder`, `decoder`, `_init_weights()` (He/Kaiming for ReLU/LeakyReLU/ELU, Xavier for Tanh), `encode(x)` (returns bottleneck code `z`), `decode(z)`, `forward(x)`, `count_parameters()`, and `get_config()`. Enforces `latent_dim < d_in` in `__init__`.
- **`DenoisingAE(UndercompleteAE)`**: Subclasses `UndercompleteAE` adding `noise_sigma` and `forward_noisy(x)` which injects `N(0, σ²I)` noise before `encode()` during training, while `forward(x)` runs cleanly with no noise for evaluation/inference.
- **`compute_reconstruction_mse(model, X, batch_size, device)`**: Runs inference in `eval()` + `torch.no_grad()` mode and returns a 1-D `float32` NumPy array of per-sample MSE (`mean((x - x̂)²)` over the 31 features). Higher = more anomalous.
- **`compute_per_feature_error(model, X, ...)`**: Returns shape `(n_samples, d_in)` squared errors `(x - x̂)²` for explaining which coordinates drove an alert.
- **`encode_dataset(model, X, batch_size, device)`**: Batched inference wrapper calling `model.encode()` for downstream consumers (Apoorva's latent GMM / Prachee's DAGMM-lite), preserving exact transaction ordering and returning `(n_samples, latent_dim)` `float32`.
- **`load_autoencoder(weights_path, config_path, device)`**: Reads `<model>_config.json`, instantiates the exact `UndercompleteAE` or `DenoisingAE`, loads `weights_only=True` state dict, and sets `.eval()`.

### `src/train_ae.py`
- **`_enable_determinism(seed)`**: Locks Python, NumPy, and PyTorch CPU/CUDA seeds plus deterministic cuDNN flags.
- **`train_autoencoder(model, X_train_legit, X_val_legit, ...)`**: Trains strictly on `X_train_legit` (`Class==0`) using a `DataLoader(shuffle=True)` for training only. Evaluates MSE on `X_val_legit` (`Class==0` only) after every epoch for `ReduceLROnPlateau` and early stopping (`patience=10`), restores the best checkpoint, and logs gradient norms and NaN checks.
- **`hyperparameter_search(...)`**: Evaluates a stratified sample of 30 configurations (20 AE + 10 DAE) over `latent_dim`, `hidden`, `dropout`, `weight_decay`, `act`, and `noise_sigma`. Ranks by `val PR-AUC` (pre-committed selection criterion; test set never touched).
- **`five_seed_experiment(...)`**: Trains the selected AE and DAE across 5 seeds (`42, 123, 456, 789, 1024`), reports `mean ± std` of PR-AUC, ROC-AUC, and Recall@FPR `{0.1%, 0.5%, 1.0%}`, and returns the median-seed model/scores (by val PR-AUC) as the official scores.
- **`regularization_study(...)` & `plot_regularization_gap(...)`**: Ablates single regularizers (`no_regularization`, `+weight_decay`, `+dropout`, `+input_noise_DAE`, `+early_stopping`, `+batchnorm`, `+L1_sparse_latent`, `all_combined`) and plots train-legit vs val-legit MSE and val PR-AUC.
- **`linear_ae_vs_pca_check(...)`**: Verifies Baldi & Hornik (1989) equivalence by comparing a linear AE (`hidden=[]`, `act='linear'`) against `sklearn.decomposition.PCA` at `k ∈ {8, 16}`.
- **`top20_fp_vs_tp_analysis(...)`**: Extracts the top-20 false positives (highest-MSE legit rows) and top-20 caught frauds (highest-MSE fraud rows) on validation and identifies the top-3 dominating features per sample and overall.
- **`run_phase2_ae()`**: End-to-end orchestrator that runs HP search, 5-seed AE/DAE, saves official `.npy` scores and `.pt` models, runs regularization study, diagnostics, `base` vs `engineered` comparison, timing, and technical validation.

---

## 3. Core Viva Questions & Answers

1. **Why must the bottleneck be undercomplete (`latent_dim < d_in`)?**  
   If `latent_dim >= d_in` without additional constraints, an MLP autoencoder can learn the trivial identity mapping $g(f(\mathbf{x})) = \mathbf{x}$. Then both legitimate transactions and frauds reconstruct with near-zero MSE, destroying anomaly separation.

2. **Why train ONLY on legitimate transactions, and why no SMOTE?**  
   - Real-world payment fraud labels arrive 30–60 days late (chargeback verification lag), so one-class normality modeling does not require fraud labels at training time.  
   - Including fraud in training lets the AE allocate capacity to reconstructing fraud patterns, reducing reconstruction contrast.  
   - SMOTE interpolates linearly between minority points, assuming fraud forms a convex cluster (false: fraud modes are disjoint) and introducing label leakage.

3. **What does early stopping monitor, and why doesn't it leak?**  
   Early stopping monitors MSE on `X_val_legit` (`56,904` legitimate validation rows — all `57` validation frauds are excluded). Because zero fraud labels or fraud rows enter the stopping criterion, training remains 100% unsupervised. Validation fraud labels are used only post-hoc to select hyperparameters (`val PR-AUC`) and freeze thresholds.

4. **What is the mathematical relationship between Linear AE and PCA?**  
   Minimizing $\mathbb{E}\|\mathbf{x} - \mathbf{W}_2\mathbf{W}_1\mathbf{x}\|_2^2$ with rank-$k$ bottleneck has the same subspace global minimum as projecting onto the top-$k$ eigenvectors of the covariance matrix (Baldi & Hornik, 1989 `[VERIFY]`). Empirically verified: at $k=8$, linear AE val-legit MSE is `0.9031` vs PCA `0.9122` (score Pearson $r = 0.9933$).

5. **How does a Denoising Autoencoder (DAE) work and what did $\sigma$ do?**  
   During training we corrupt inputs $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\varepsilon}$ and minimize $\|\mathbf{x} - g(f(\tilde{\mathbf{x}}))\|_2^2$ against the clean target $\mathbf{x}$. Vincent (2011 `[VERIFY]`) proved this approximates score matching ($\nabla_{\mathbf{x}} \log p(\mathbf{x})$), pulling noisy points back to the normal manifold. On test, DAE ($\sigma=0.2$) improved median test PR-AUC from `0.3574` (AE) to `0.4818` and `Recall@0.5%` from `0.6800` to `0.7733` with much tighter seed stability (`0.7653 ± 0.0160`).

6. **When does MSE reconstruction error fail as an anomaly score?**  
   - **On-manifold anomalies:** A fraud lying close to the learned $k$-dimensional manifold has low $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ even if it sits in a very low-density region on the manifold — this motivates Apoorva's Latent-GMM ($H_1$) and Prachee's hybrids ($H_2, H_3$).  
   - **High-variance dominance:** Unscaled high-variance coordinates (raw `Amount`) dominate unweighted MSE; train-legit `StandardScaler` resolves this.  
   - **Tail-outlier sensitivity:** Even one extreme legitimate outlier in `V3` or `V8` can sit at the top of the MSE ranking and cause high PR-AUC variance across seeds (`±0.13`).

7. **What regularizers did you test and what did each do?**  
   - **`+L1_sparse_latent (1e-3)`** gave the best val PR-AUC (`0.3017`), highest `Recall@0.1%` (`0.6316`), and smallest train–val gap (`0.1342`).  
   - **`+dropout (0.2)`** severely degraded performance (`val PR-AUC = 0.0462`, `val MSE = 0.9144`) because dropping 20% of units inside a small `[24, 16]` bottleneck causes underfitting.  
   - **`+batchnorm`** raised `Recall@0.5%` (`0.7368`) but lowered PR-AUC (`0.1840`).  
   - **`engineered` features** (pruning `V22, V23, V25` + adding geometric $V$-norms) boosted DAE test PR-AUC to `0.5213` (val `0.5617`) and AE test PR-AUC to `0.3724`.

---

## 4. `encode()` Interface Specification (for Apoorva & Prachee)
- **Model loading:**
  ```python
  from src.autoencoder import load_autoencoder, encode_dataset
  model = load_autoencoder("results/models/ae_best.pt", "results/models/ae_config.json")
  ```
  (Also available: `ae_dae_best.pt` and `ae_latent_k{2,4,8,16}.pt` for Apoorva's latent-dimension ablation.)
- **Input shape & scaling:** `np.ndarray` of shape `(N, 31)`, `float32`, standardized via `get_features("base")` (scaler fit on train-legit only).
- **Latent dimension:** `model.latent_dim` (`12` for `ae_best.pt`, `16` for `ae_dae_best.pt`, or `{2, 4, 8, 16}` for `ae_latent_k*.pt`).
- **Output shape & dtype:** `np.ndarray` of shape `(N, latent_dim)`, `dtype=np.float32`.
- **Transaction ordering:** Strictly preserves input row order (chronological order matching `data/processed/{train,val,test}.csv` and `split_manifest.json`).

