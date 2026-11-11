"""
src/train_ae.py
===============
Training pipeline for Deep Autoencoders — Team T033 Phase 2.
Author: Vani Rudra (Task P2_03)

Implements:
    - Legitimate-only training with early stopping on val-legit MSE.
    - Hyperparameter search (random/grid over latent_dim, hidden, dropout, etc.).
    - 5-seed reproducibility experiments.
    - Regularization ablation study.
    - Score generation and saving.
    - Timing and diagnostics.

IMPORTANT LEAKAGE RULES:
    1. Train ONLY on legitimate training transactions (Class==0).
    2. Early stopping uses val-legit MSE loss ONLY (no fraud labels in stopping criterion).
    3. Fraud labels are used ONLY for model selection via val PR-AUC (never test).
    4. Test set is never used for model or threshold selection.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import (
    UndercompleteAE,
    DenoisingAE,
    compute_reconstruction_mse,
    compute_per_feature_error,
    encode_dataset,
)
from src.config import CFG, get_repo_root
from src.metrics import pr_auc, roc_auc, recall_at_fpr, evaluate_scores
from src.utils import get_logger, set_seed, save_scores, save_table, save_fig

_logger = get_logger(__name__)
_ROOT = get_repo_root()
_MODELS_DIR = _ROOT / CFG["paths"]["models"]
_TABLES_DIR = _ROOT / CFG["paths"]["tables"]
_FIGURES_DIR = _ROOT / CFG["paths"]["figures"]
_SCORES_DIR = _ROOT / CFG["paths"]["scores"]


def _ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [_MODELS_DIR, _TABLES_DIR, _FIGURES_DIR, _SCORES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def _enable_determinism(seed: int) -> None:
    """Set all RNG seeds and PyTorch deterministic flags."""
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_autoencoder(
    model: UndercompleteAE,
    X_train_legit: np.ndarray,
    X_val_legit: np.ndarray,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
    is_dae: bool = False,
    l1_latent_lambda: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Train an autoencoder on legitimate transactions only.

    Early stopping monitors val-legit MSE loss (NOT any metric that uses
    fraud labels). This is critical: val-legit MSE tells us when the model
    starts overfitting to the training distribution without leaking any
    information about what fraud looks like.

    Parameters
    ----------
    model : UndercompleteAE or DenoisingAE
        The model to train (will be modified in place).
    X_train_legit : np.ndarray, shape (n_train_legit, d_in)
        Legitimate training features ONLY.
    X_val_legit : np.ndarray, shape (n_val_legit, d_in)
        Legitimate validation features for early stopping.
    lr : float
        Learning rate for Adam.
    weight_decay : float
        L2 regularization (Adam weight_decay parameter).
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (epochs without val-legit loss improvement).
    device : str
        'cpu', 'cuda', or 'mps'.
    is_dae : bool
        If True, use denoising forward pass (add noise during training).
    l1_latent_lambda : float
        Optional L1 sparsity penalty weight on latent code z.
    verbose : bool
        Print per-epoch logs.

    Returns
    -------
    dict
        Training history: train_losses, val_losses, grad_norms, nan_detected,
        best_epoch, best_val_loss, train_time_seconds, avg_epoch_time_seconds,
        epochs_trained.
    """
    model = model.to(device)

    # DataLoader — shuffle within training batches only
    train_ds = TensorDataset(torch.tensor(X_train_legit, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_legit, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    grad_norms = []
    nan_detected = False

    t0 = time.time()

    for epoch in range(max_epochs):
        # ── Training phase ──
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n_batches = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            if is_dae and isinstance(model, DenoisingAE):
                x_noisy = x_batch + torch.randn_like(x_batch) * model.noise_sigma
                z = model.encode(x_noisy)
                x_hat = model.decode(z)
            else:
                z = model.encode(x_batch)
                x_hat = model.decode(z)

            loss = criterion(x_hat, x_batch)  # Compare to CLEAN input
            if l1_latent_lambda > 0.0:
                loss = loss + l1_latent_lambda * z.abs().mean()

            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                _logger.error("NaN/Inf loss detected at epoch %d!", epoch + 1)
                break

            loss.backward()

            # Gradient clipping & norm tracking for technical validation
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            epoch_grad_norm += float(total_norm.item())
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if nan_detected or n_batches == 0:
            break

        avg_train_loss = epoch_loss / n_batches
        avg_grad_norm = epoch_grad_norm / n_batches
        grad_norms.append(avg_grad_norm)

        # ── Validation phase (legit-only MSE) ──
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                x_hat = model(x_batch)  # No noise at validation
                loss = criterion(x_hat, x_batch)
                val_loss += float(loss.item())
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if verbose and (epoch + 1) % 5 == 0:
            _logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  grad_norm=%.4f  lr=%.2e",
                epoch + 1, max_epochs, avg_train_loss, avg_val_loss,
                avg_grad_norm, optimizer.param_groups[0]["lr"],
            )

        # ── Early stopping ──
        if avg_val_loss < best_val_loss - 1e-7:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                _logger.info(
                    "Early stopping at epoch %d (patience=%d). Best val_loss=%.6f",
                    epoch + 1, patience, best_val_loss,
                )
                break

    train_time = time.time() - t0
    epochs_trained = len(train_losses)

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_epoch = epochs_trained - epochs_without_improvement

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "grad_norms": grad_norms,
        "nan_detected": nan_detected,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_time_seconds": round(train_time, 2),
        "avg_epoch_time_seconds": round(train_time / max(1, epochs_trained), 4),
        "epochs_trained": epochs_trained,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameter search
# ═══════════════════════════════════════════════════════════════════════════════

def hyperparameter_search(
    X_train_legit: np.ndarray,
    X_val_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
    max_configs: int = 30,
) -> pd.DataFrame:
    """
    Grid/random search over autoencoder hyperparameters.

    Selection criterion (fixed BEFORE running): **val PR-AUC**.
    Val labels are used for model selection only — never test.
    We also report val-legit reconstruction loss for transparency.

    Parameters
    ----------
    X_train_legit : training legitimate features.
    X_val_legit : validation legitimate features (for early stopping).
    X_val : full validation features (for PR-AUC evaluation).
    y_val : validation labels (for PR-AUC evaluation only).
    device : compute device.
    max_configs : maximum number of configurations to try.

    Returns
    -------
    pd.DataFrame with one row per configuration, sorted by val PR-AUC descending.
    """
    d_in = X_train_legit.shape[1]
    seed = int(CFG.get("seed", 42))

    # Define search space covering all required axes
    ae_pool = []
    for latent_dim in [2, 4, 8, 12, 16]:
        for hidden in [[24, 16], [32, 16], [24, 20]]:
            for dropout in [0.0, 0.1, 0.2]:
                for wd in [0.0, 1e-5, 1e-4]:
                    for act in ["relu", "leaky_relu", "tanh"]:
                        ae_pool.append({
                            "latent_dim": latent_dim,
                            "hidden": hidden,
                            "dropout": dropout,
                            "weight_decay": wd,
                            "act": act,
                            "is_dae": False,
                            "noise_sigma": 0.0,
                        })

    dae_pool = []
    for latent_dim in [4, 8, 12, 16]:
        for sigma in [0.05, 0.1, 0.2, 0.5]:
            for act in ["relu", "leaky_relu", "tanh"]:
                dae_pool.append({
                    "latent_dim": latent_dim,
                    "hidden": [24, 16],
                    "dropout": 0.0,
                    "weight_decay": 1e-5,
                    "act": act,
                    "is_dae": True,
                    "noise_sigma": sigma,
                })

    # Stratified sample within budget (<= max_configs, default 30: 20 AE + 10 DAE)
    rng = np.random.default_rng(seed)
    n_dae = min(len(dae_pool), max(1, max_configs // 3))
    n_ae = min(len(ae_pool), max_configs - n_dae)

    ae_idx = sorted(rng.choice(len(ae_pool), n_ae, replace=False))
    dae_idx = sorted(rng.choice(len(dae_pool), n_dae, replace=False))
    configs = [ae_pool[i] for i in ae_idx] + [dae_pool[i] for i in dae_idx]

    _logger.info("Running hyperparameter search: %d configurations", len(configs))

    results = []
    for i, cfg_hp in enumerate(configs):
        _enable_determinism(seed)

        if cfg_hp["is_dae"]:
            model = DenoisingAE(
                d_in=d_in,
                latent_dim=cfg_hp["latent_dim"],
                hidden=cfg_hp["hidden"],
                act=cfg_hp["act"],
                dropout=cfg_hp["dropout"],
                noise_sigma=cfg_hp["noise_sigma"],
            )
        else:
            model = UndercompleteAE(
                d_in=d_in,
                latent_dim=cfg_hp["latent_dim"],
                hidden=cfg_hp["hidden"],
                act=cfg_hp["act"],
                dropout=cfg_hp["dropout"],
            )

        history = train_autoencoder(
            model, X_train_legit, X_val_legit,
            lr=1e-3, weight_decay=cfg_hp["weight_decay"],
            batch_size=512, max_epochs=100, patience=10,
            device=device, is_dae=cfg_hp["is_dae"], verbose=False,
        )

        # Evaluate on val
        val_scores = compute_reconstruction_mse(model, X_val, device=device)
        val_pr_auc = pr_auc(y_val, val_scores)
        val_roc_auc = roc_auc(y_val, val_scores)
        val_recall_001 = recall_at_fpr(y_val, val_scores, 0.001)
        val_recall_005 = recall_at_fpr(y_val, val_scores, 0.005)

        row = {
            "config_id": i,
            "latent_dim": cfg_hp["latent_dim"],
            "hidden": str(cfg_hp["hidden"]),
            "dropout": cfg_hp["dropout"],
            "weight_decay": cfg_hp["weight_decay"],
            "act": cfg_hp["act"],
            "is_dae": cfg_hp["is_dae"],
            "noise_sigma": cfg_hp["noise_sigma"],
            "n_params": model.count_parameters(),
            "best_epoch": history["best_epoch"],
            "best_val_legit_loss": history["best_val_loss"],
            "val_pr_auc": round(val_pr_auc, 5),
            "val_roc_auc": round(val_roc_auc, 5),
            "val_recall@0.1%": round(val_recall_001, 5),
            "val_recall@0.5%": round(val_recall_005, 5),
            "train_time_sec": history["train_time_seconds"],
        }
        results.append(row)

        _logger.info(
            "Config %2d/%d  latent=%d hidden=%s  PR-AUC=%.4f  val_loss=%.6f  time=%.1fs",
            i + 1, len(configs), cfg_hp["latent_dim"], cfg_hp["hidden"],
            val_pr_auc, history["best_val_loss"], history["train_time_seconds"],
        )

    df = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5-seed experiment
# ═══════════════════════════════════════════════════════════════════════════════

def five_seed_experiment(
    model_cls,
    model_kwargs: dict,
    X_train_legit: np.ndarray,
    X_val_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    is_dae: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seeds: tuple[int, ...] = (42, 123, 456, 789, 1024),
) -> tuple[pd.DataFrame, dict]:
    """
    Train the chosen model with 5 different seeds to report mean ± std.

    Returns the seed results table and the scores from the median-seed run
    (used as official scores).

    Parameters
    ----------
    model_cls : class (UndercompleteAE or DenoisingAE)
    model_kwargs : dict of constructor arguments
    seeds : tuple of int, 5 random seeds

    Returns
    -------
    results_df : pd.DataFrame with one row per seed
    median_scores : dict with 'val_scores', 'test_scores', 'seed', 'model_state'
    """
    results = []
    all_val_scores = {}
    all_test_scores = {}
    all_states = {}

    for seed_val in seeds:
        _enable_determinism(seed_val)

        model = model_cls(**model_kwargs)
        history = train_autoencoder(
            model, X_train_legit, X_val_legit,
            lr=lr, weight_decay=weight_decay,
            batch_size=512, max_epochs=100, patience=10,
            device=device, is_dae=is_dae, verbose=False,
        )

        val_scores = compute_reconstruction_mse(model, X_val, device=device)
        test_scores = compute_reconstruction_mse(model, X_test, device=device)

        val_prauc = pr_auc(y_val, val_scores)
        test_prauc = pr_auc(y_test, test_scores)
        val_recall_001 = recall_at_fpr(y_val, val_scores, 0.001)
        val_recall_005 = recall_at_fpr(y_val, val_scores, 0.005)
        val_recall_010 = recall_at_fpr(y_val, val_scores, 0.010)
        test_recall_001 = recall_at_fpr(y_test, test_scores, 0.001)
        test_recall_005 = recall_at_fpr(y_test, test_scores, 0.005)
        test_recall_010 = recall_at_fpr(y_test, test_scores, 0.010)

        results.append({
            "seed": seed_val,
            "val_pr_auc": val_prauc,
            "test_pr_auc": test_prauc,
            "val_roc_auc": roc_auc(y_val, val_scores),
            "test_roc_auc": roc_auc(y_test, test_scores),
            "val_recall@0.1%": val_recall_001,
            "test_recall@0.1%": test_recall_001,
            "val_recall@0.5%": val_recall_005,
            "test_recall@0.5%": test_recall_005,
            "val_recall@1.0%": val_recall_010,
            "test_recall@1.0%": test_recall_010,
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "train_time_sec": history["train_time_seconds"],
        })
        all_val_scores[seed_val] = val_scores
        all_test_scores[seed_val] = test_scores
        all_states[seed_val] = copy.deepcopy(model.state_dict())

        _logger.info(
            "Seed %d: val_PR-AUC=%.4f  test_PR-AUC=%.4f  epochs=%d  time=%.1fs",
            seed_val, val_prauc, test_prauc,
            history["best_epoch"], history["train_time_seconds"],
        )

    df = pd.DataFrame(results)

    # Add mean ± std summary row
    summary = {"seed": "mean±std"}
    for col in df.columns:
        if col == "seed":
            continue
        vals = df[col].values.astype(float)
        summary[col] = f"{vals.mean():.5f}±{vals.std():.5f}"
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    # Select median seed by val PR-AUC
    numeric_df = df.iloc[:-1]  # exclude summary row
    val_praucs = numeric_df["val_pr_auc"].astype(float).values
    median_idx = np.argsort(val_praucs)[len(val_praucs) // 2]
    median_seed = int(numeric_df.iloc[median_idx]["seed"])

    _logger.info("Median seed (by val PR-AUC): %d", median_seed)

    median_scores = {
        "val_scores": all_val_scores[median_seed],
        "test_scores": all_test_scores[median_seed],
        "seed": median_seed,
        "model_state": all_states[median_seed],
    }

    return df, median_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Regularization study
# ═══════════════════════════════════════════════════════════════════════════════

def regularization_study(
    X_train_legit: np.ndarray,
    X_val_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    d_in: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Regularization ablation: one factor at a time from baseline config.

    Rubric: DL Dataset & Regularization — table showing val PR-AUC for
    each regularization configuration.

    Configurations tested:
    1. No regularization (baseline: no dropout, no weight_decay, no early stopping cap)
    2. + weight_decay only
    3. + dropout only
    4. + input noise (DAE)
    5. + early stopping (patience=10)
    6. + batch normalization
    7. All combined (final config)
    """
    seed = int(CFG.get("seed", 42))

    configs = [
        {
            "name": "no_regularization",
            "cls": UndercompleteAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.0, "batchnorm": False},
            "train_kw": {"weight_decay": 0.0, "patience": 200, "max_epochs": 100},
            "is_dae": False,
        },
        {
            "name": "+weight_decay",
            "cls": UndercompleteAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.0, "batchnorm": False},
            "train_kw": {"weight_decay": 1e-5, "patience": 200, "max_epochs": 100},
            "is_dae": False,
        },
        {
            "name": "+dropout",
            "cls": UndercompleteAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.2, "batchnorm": False},
            "train_kw": {"weight_decay": 0.0, "patience": 200, "max_epochs": 100},
            "is_dae": False,
        },
        {
            "name": "+input_noise_DAE(σ=0.1)",
            "cls": DenoisingAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.0, "batchnorm": False,
                       "noise_sigma": 0.1},
            "train_kw": {"weight_decay": 0.0, "patience": 200, "max_epochs": 100},
            "is_dae": True,
        },
        {
            "name": "+early_stopping(p=10)",
            "cls": UndercompleteAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.0, "batchnorm": False},
            "train_kw": {"weight_decay": 0.0, "patience": 10, "max_epochs": 100},
            "is_dae": False,
        },
        {
            "name": "+batchnorm",
            "cls": UndercompleteAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.0, "batchnorm": True},
            "train_kw": {"weight_decay": 0.0, "patience": 200, "max_epochs": 100},
            "is_dae": False,
        },
        {
            "name": "+L1_sparse_latent(1e-3)",
            "cls": UndercompleteAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.0, "batchnorm": False},
            "train_kw": {"weight_decay": 0.0, "patience": 200, "max_epochs": 100,
                         "l1_latent_lambda": 1e-3},
            "is_dae": False,
        },
        {
            "name": "all_combined",
            "cls": DenoisingAE,
            "kwargs": {"d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
                       "act": "relu", "dropout": 0.1, "batchnorm": False,
                       "noise_sigma": 0.1},
            "train_kw": {"weight_decay": 1e-5, "patience": 10, "max_epochs": 100},
            "is_dae": True,
        },
    ]

    results = []
    for cfg_item in configs:
        _enable_determinism(seed)
        model = cfg_item["cls"](**cfg_item["kwargs"])
        history = train_autoencoder(
            model, X_train_legit, X_val_legit,
            lr=1e-3, device=device,
            is_dae=cfg_item["is_dae"],
            verbose=False,
            **cfg_item["train_kw"],
        )
        val_scores = compute_reconstruction_mse(model, X_val, device=device)

        # Compute train reconstruction loss for over/under-fitting evidence
        train_scores = compute_reconstruction_mse(model, X_train_legit, device=device)
        val_legit_eval = compute_reconstruction_mse(model, X_val_legit, device=device)

        results.append({
            "config": cfg_item["name"],
            "val_pr_auc": round(pr_auc(y_val, val_scores), 5),
            "val_roc_auc": round(roc_auc(y_val, val_scores), 5),
            "val_recall@0.1%FPR": round(recall_at_fpr(y_val, val_scores, 0.001), 5),
            "val_recall@0.5%FPR": round(recall_at_fpr(y_val, val_scores, 0.005), 5),
            "train_legit_mse": round(float(train_scores.mean()), 6),
            "val_legit_mse": round(float(val_legit_eval.mean()), 6),
            "train_val_gap": round(float(val_legit_eval.mean()) - float(train_scores.mean()), 6),
            "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"],
            "n_params": model.count_parameters(),
        })
        _logger.info(
            "Reg study [%s]: PR-AUC=%.4f  train_loss=%.6f  val_loss=%.6f  gap=%.6f",
            cfg_item["name"], results[-1]["val_pr_auc"],
            results[-1]["train_legit_mse"], results[-1]["val_legit_mse"],
            results[-1]["train_val_gap"],
        )

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# Additional P2_03 Diagnostics: PCA Sanity Check, Top-20 FP vs TP, Gap Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_regularization_gap(
    reg_df: pd.DataFrame,
    save_name: str = "ae_regularization_gap",
) -> plt.Figure:
    """Plot over/under-fit evidence (train vs val-legit MSE and val PR-AUC)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(reg_df))
    w = 0.35
    ax1.bar(x - w / 2, reg_df["train_legit_mse"], width=w, label="Train-legit MSE", color="#2196F3")
    ax1.bar(x + w / 2, reg_df["val_legit_mse"], width=w, label="Val-legit MSE", color="#FF9800")
    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_df["config"], rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("Train vs Val-Legit Reconstruction Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.barh(reg_df["config"], reg_df["val_pr_auc"], color="#4CAF50", alpha=0.85)
    ax2.set_xlabel("Validation PR-AUC")
    ax2.set_title("Regularization Effect on Val PR-AUC")
    ax2.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    plt.close(fig)
    return fig


def linear_ae_vs_pca_check(
    X_train_legit: np.ndarray,
    X_val_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ks: tuple[int, ...] = (8, 16),
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Sanity check: compare a single-hidden-layer linear AE against sklearn PCA
    at the same bottleneck dimension k.

    Theory (Baldi & Hornik 1989): a linear AE trained with MSE spans the
    same subspace as the top-k principal components of PCA.
    """
    from sklearn.decomposition import PCA

    d_in = X_train_legit.shape[1]
    seed = int(CFG.get("seed", 42))
    rows = []

    for k in ks:
        # 1. Exact PCA
        pca = PCA(n_components=k, random_state=seed)
        pca.fit(X_train_legit)
        X_val_pca_hat = pca.inverse_transform(pca.transform(X_val))
        pca_mse_all = np.mean((X_val - X_val_pca_hat) ** 2, axis=1)
        pca_val_legit_mse = float(pca_mse_all[y_val == 0].mean())
        pca_val_prauc = pr_auc(y_val, pca_mse_all)

        # 2. Linear AE (hidden=[], act='linear')
        _enable_determinism(seed)
        lin_ae = UndercompleteAE(d_in=d_in, latent_dim=k, hidden=[], act="linear")
        train_autoencoder(
            lin_ae, X_train_legit, X_val_legit,
            lr=1e-3, weight_decay=0.0, batch_size=512,
            max_epochs=60, patience=10, device=device, verbose=False,
        )
        ae_mse_all = compute_reconstruction_mse(lin_ae, X_val, device=device)
        ae_val_legit_mse = float(ae_mse_all[y_val == 0].mean())
        ae_val_prauc = pr_auc(y_val, ae_mse_all)

        corr = float(np.corrcoef(pca_mse_all, ae_mse_all)[0, 1])

        rows.append({
            "k": k,
            "pca_val_legit_mse": round(pca_val_legit_mse, 6),
            "linear_ae_val_legit_mse": round(ae_val_legit_mse, 6),
            "mse_abs_diff": round(abs(pca_val_legit_mse - ae_val_legit_mse), 6),
            "pca_val_pr_auc": round(pca_val_prauc, 5),
            "linear_ae_val_pr_auc": round(ae_val_prauc, 5),
            "score_pearson_corr": round(corr, 5),
        })
        _logger.info(
            "Linear-AE vs PCA (k=%d): PCA_MSE=%.6f, LinAE_MSE=%.6f, corr=%.4f",
            k, pca_val_legit_mse, ae_val_legit_mse, corr,
        )

    # Also compare with saved Phase 1 PCA baseline scores if available
    p1_pca_path = _SCORES_DIR / "pca_val.npy"
    if p1_pca_path.exists():
        p1_pca = np.load(p1_pca_path)
        _logger.info("Phase 1 PCA baseline val PR-AUC = %.5f", pr_auc(y_val, p1_pca))

    return pd.DataFrame(rows)


def top20_fp_vs_tp_analysis(
    model: UndercompleteAE,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    feature_names: list[str],
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare top-20 false positives (highest MSE among legit) vs
    top-20 caught frauds (highest MSE among frauds) and identify
    which features dominate their reconstruction error.
    """
    feat_err = compute_per_feature_error(model, X_val, device=device)

    legit_idx = np.where(y_val == 0)[0]
    fraud_idx = np.where(y_val == 1)[0]

    top_fp_idx = legit_idx[np.argsort(val_scores[legit_idx])[::-1][:20]]
    top_tp_idx = fraud_idx[np.argsort(val_scores[fraud_idx])[::-1][:20]]

    rows = []
    for group_name, idxs in [("top20_FP_legit", top_fp_idx), ("top20_TP_fraud", top_tp_idx)]:
        for rank, idx in enumerate(idxs, start=1):
            row_err = feat_err[idx]
            top3 = np.argsort(row_err)[::-1][:3]
            total_err = float(row_err.sum()) + 1e-12
            rows.append({
                "group": group_name,
                "rank": rank,
                "val_row_idx": int(idx),
                "mse_score": round(float(val_scores[idx]), 5),
                "top1_feature": feature_names[top3[0]],
                "top1_share": round(float(row_err[top3[0]] / total_err), 4),
                "top2_feature": feature_names[top3[1]],
                "top2_share": round(float(row_err[top3[1]] / total_err), 4),
                "top3_feature": feature_names[top3[2]],
                "top3_share": round(float(row_err[top3[2]] / total_err), 4),
            })

    detail_df = pd.DataFrame(rows)

    # Aggregate mean per-feature error for top-20 FP vs top-20 TP
    fp_mean = feat_err[top_fp_idx].mean(axis=0)
    tp_mean = feat_err[top_tp_idx].mean(axis=0)
    dom_df = pd.DataFrame({
        "feature": feature_names,
        "top20_fp_mean_sqerr": fp_mean,
        "top20_tp_mean_sqerr": tp_mean,
        "tp_to_fp_ratio": tp_mean / (fp_mean + 1e-10),
    }).sort_values("top20_tp_mean_sqerr", ascending=False)

    return detail_df, dom_df



# ═══════════════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(
    history: dict,
    title: str = "AE Training Curves",
    save_name: str | None = None,
) -> plt.Figure:
    """Plot training vs val-legit loss with early-stopping epoch overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_losses"]) + 1)
    ax.plot(epochs, history["train_losses"], label="Train (legit) MSE", color="#2196F3")
    ax.plot(epochs, history["val_losses"], label="Val (legit) MSE", color="#F44336")
    ax.axvline(x=history["best_epoch"], color="green", linestyle="--",
               label=f"Best epoch = {history['best_epoch']}", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_error_histogram(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    title: str = "Reconstruction Error Distribution",
    save_name: str | None = None,
) -> plt.Figure:
    """Histogram of reconstruction error: legit vs fraud."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(val_scores[y_val == 0], bins=80, alpha=0.6, label="Legitimate",
            color="#2196F3", density=True)
    ax.hist(val_scores[y_val == 1], bins=80, alpha=0.7, label="Fraud",
            color="#F44336", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("Per-sample MSE")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_latent_scatter(
    codes: np.ndarray,
    labels: np.ndarray,
    title: str = "Latent Space (2D)",
    save_name: str | None = None,
) -> plt.Figure:
    """2D scatter of latent codes colored by class."""
    if codes.shape[1] > 2:
        from sklearn.decomposition import PCA
        codes_2d = PCA(n_components=2).fit_transform(codes)
    else:
        codes_2d = codes

    fig, ax = plt.subplots(figsize=(8, 6))
    legit = labels == 0
    fraud = labels == 1
    ax.scatter(codes_2d[legit, 0], codes_2d[legit, 1], s=1, alpha=0.3,
               c="#2196F3", label="Legitimate")
    ax.scatter(codes_2d[fraud, 0], codes_2d[fraud, 1], s=20, alpha=0.8,
               c="#F44336", label="Fraud", marker="x")
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2_ae():
    """
    Complete Phase 2 autoencoder pipeline:
    1. Load data
    2. Hyperparameter search
    3. 5-seed experiment for chosen AE and DAE
    4. Regularization study
    5. Save official scores, models, tables, figures
    6. Timing and diagnostics
    """
    from src.data import get_features

    _ensure_dirs()
    device = _get_device()
    _logger.info("Using device: %s", device)

    # ── 1. Load data ──
    _logger.info("Loading features...")
    splits = get_features("base")
    X_train_legit = splits.X_train_legit
    X_val = splits.X_val
    y_val = splits.y_val
    X_test = splits.X_test
    y_test = splits.y_test
    d_in = X_train_legit.shape[1]

    # Val-legit for early stopping (no fraud labels in stopping criterion)
    X_val_legit = X_val[y_val == 0]

    _logger.info(
        "Data loaded: d_in=%d, train_legit=%d, val=%d (legit=%d, fraud=%d), test=%d",
        d_in, len(X_train_legit), len(X_val), len(X_val_legit),
        y_val.sum(), len(X_test),
    )

    # ── 2. Hyperparameter search ──
    _logger.info("=" * 70)
    _logger.info("HYPERPARAMETER SEARCH")
    _logger.info("=" * 70)

    hp_df = hyperparameter_search(
        X_train_legit, X_val_legit, X_val, y_val,
        device=device, max_configs=30,
    )
    save_table(hp_df, "ae_hyperparam_search")
    _logger.info("Best AE config:\n%s", hp_df.iloc[0].to_string())

    # Extract best AE (non-DAE) config
    ae_configs = hp_df[hp_df["is_dae"] == False]  # noqa: E712
    best_ae = ae_configs.iloc[0]
    ae_kwargs = {
        "d_in": d_in,
        "latent_dim": int(best_ae["latent_dim"]),
        "hidden": eval(best_ae["hidden"]),
        "act": best_ae["act"],
        "dropout": float(best_ae["dropout"]),
    }
    best_ae_wd = float(best_ae["weight_decay"])

    # Extract best DAE config
    dae_configs = hp_df[hp_df["is_dae"] == True]  # noqa: E712
    if len(dae_configs) > 0:
        best_dae = dae_configs.iloc[0]
        dae_kwargs = {
            "d_in": d_in,
            "latent_dim": int(best_dae["latent_dim"]),
            "hidden": eval(best_dae["hidden"]),
            "act": best_dae["act"],
            "dropout": float(best_dae["dropout"]),
            "noise_sigma": float(best_dae["noise_sigma"]),
        }
        best_dae_wd = float(best_dae["weight_decay"])
    else:
        # Default DAE
        dae_kwargs = {
            "d_in": d_in, "latent_dim": 8, "hidden": [24, 16],
            "act": "relu", "dropout": 0.0, "noise_sigma": 0.1,
        }
        best_dae_wd = 1e-5

    # ── 3. 5-seed AE experiment ──
    _logger.info("=" * 70)
    _logger.info("5-SEED EXPERIMENT: Undercomplete AE")
    _logger.info("=" * 70)

    ae_seed_df, ae_median = five_seed_experiment(
        UndercompleteAE, ae_kwargs,
        X_train_legit, X_val_legit, X_val, y_val, X_test, y_test,
        is_dae=False, weight_decay=best_ae_wd, device=device,
    )
    save_table(ae_seed_df, "ae_5seed_results")

    # Save official AE scores (median seed)
    save_scores("ae", "val", ae_median["val_scores"])
    save_scores("ae", "test", ae_median["test_scores"])

    # Save AE model weights
    ae_model = UndercompleteAE(**ae_kwargs)
    ae_model.load_state_dict(ae_median["model_state"])
    torch.save(ae_median["model_state"], _MODELS_DIR / "ae_best.pt")
    with open(_MODELS_DIR / "ae_config.json", "w") as f:
        json.dump(ae_model.get_config(), f, indent=2)

    # ── 4. 5-seed DAE experiment ──
    _logger.info("=" * 70)
    _logger.info("5-SEED EXPERIMENT: Denoising AE")
    _logger.info("=" * 70)

    dae_seed_df, dae_median = five_seed_experiment(
        DenoisingAE, dae_kwargs,
        X_train_legit, X_val_legit, X_val, y_val, X_test, y_test,
        is_dae=True, weight_decay=best_dae_wd, device=device,
    )
    save_table(dae_seed_df, "ae_dae_5seed_results")

    # Save official DAE scores (median seed)
    save_scores("ae_dae", "val", dae_median["val_scores"])
    save_scores("ae_dae", "test", dae_median["test_scores"])

    # Save DAE model weights
    dae_model = DenoisingAE(**dae_kwargs)
    dae_model.load_state_dict(dae_median["model_state"])
    torch.save(dae_median["model_state"], _MODELS_DIR / "ae_dae_best.pt")
    with open(_MODELS_DIR / "ae_dae_config.json", "w") as f:
        json.dump(dae_model.get_config(), f, indent=2)

    # ── 5. Evaluation metrics tables ──
    _logger.info("=" * 70)
    _logger.info("EVALUATION METRICS")
    _logger.info("=" * 70)

    ae_metrics = evaluate_scores(y_val, ae_median["val_scores"],
                                  y_test, ae_median["test_scores"], CFG)
    save_table(ae_metrics, "ae_metrics")

    dae_metrics = evaluate_scores(y_val, dae_median["val_scores"],
                                   y_test, dae_median["test_scores"], CFG)
    save_table(dae_metrics, "ae_dae_metrics")

    # ── 6. Regularization study ──
    _logger.info("=" * 70)
    _logger.info("REGULARIZATION STUDY")
    _logger.info("=" * 70)

    reg_df = regularization_study(
        X_train_legit, X_val_legit, X_val, y_val, d_in, device=device,
    )
    save_table(reg_df, "ae_regularization_study")
    plot_regularization_gap(reg_df, "ae_regularization_gap")

    # ── 7. Learning curves and diagnostics ──
    _logger.info("=" * 70)
    _logger.info("DIAGNOSTICS & VISUALIZATIONS")
    _logger.info("=" * 70)

    # Train final AE again for learning curve plot & grad norm check
    _enable_determinism(ae_median["seed"])
    final_ae = UndercompleteAE(**ae_kwargs)
    final_history = train_autoencoder(
        final_ae, X_train_legit, X_val_legit,
        lr=1e-3, weight_decay=best_ae_wd,
        batch_size=512, max_epochs=100, patience=10,
        device=device, is_dae=False,
    )
    plot_learning_curves(final_history, "AE Learning Curves", "ae_learning_curves")

    # Train final DAE for learning curve plot & grad norm check
    _enable_determinism(dae_median["seed"])
    final_dae = DenoisingAE(**dae_kwargs)
    final_dae_history = train_autoencoder(
        final_dae, X_train_legit, X_val_legit,
        lr=1e-3, weight_decay=best_dae_wd,
        batch_size=512, max_epochs=100, patience=10,
        device=device, is_dae=True,
    )
    plot_learning_curves(final_dae_history, "DAE Learning Curves", "ae_dae_learning_curves")

    # Error histograms
    plot_error_histogram(ae_median["val_scores"], y_val,
                         "AE Reconstruction Error (Val)", "ae_error_histogram")
    plot_error_histogram(dae_median["val_scores"], y_val,
                         "DAE Reconstruction Error (Val)", "ae_dae_error_histogram")

    # Latent space scatter (chosen AE via PCA-2D if latent_dim > 2)
    ae_model.eval()
    val_codes = encode_dataset(ae_model, X_val, device=device)
    plot_latent_scatter(val_codes, y_val, "AE Latent Space (Val)", "ae_latent_scatter")

    # Train & save latent-dim ablation checkpoints (k in 2, 4, 8, 16) for Apoorva + explicit 2D scatter
    latent_ablation_rows = []
    for k_lat in (2, 4, 8, 16):
        _enable_determinism(int(CFG.get("seed", 42)))
        k_kwargs = {**ae_kwargs, "latent_dim": k_lat}
        k_model = UndercompleteAE(**k_kwargs)
        k_hist = train_autoencoder(
            k_model, X_train_legit, X_val_legit,
            lr=1e-3, weight_decay=best_ae_wd,
            batch_size=512, max_epochs=100, patience=10,
            device=device, is_dae=False, verbose=False,
        )
        k_val_scores = compute_reconstruction_mse(k_model, X_val, device=device)
        torch.save(k_model.state_dict(), _MODELS_DIR / f"ae_latent_k{k_lat}.pt")
        with open(_MODELS_DIR / f"ae_latent_k{k_lat}_config.json", "w") as f:
            json.dump(k_model.get_config(), f, indent=2)
        latent_ablation_rows.append({
            "latent_dim": k_lat,
            "n_params": k_model.count_parameters(),
            "best_val_legit_mse": round(k_hist["best_val_loss"], 6),
            "val_pr_auc": round(pr_auc(y_val, k_val_scores), 5),
            "val_recall@0.5%FPR": round(recall_at_fpr(y_val, k_val_scores, 0.005), 5),
        })
        if k_lat == 2:
            codes_2d = encode_dataset(k_model, X_val, device=device)
            plot_latent_scatter(
                codes_2d, y_val,
                "AE Latent Space (latent_dim=2, Val)",
                "ae_latent_dim2_scatter",
            )
    save_table(pd.DataFrame(latent_ablation_rows), "ae_latent_dim_sweep")

    # Linear AE vs PCA sanity check
    pca_sanity_df = linear_ae_vs_pca_check(
        X_train_legit, X_val_legit, X_val, y_val, ks=(8, 16), device=device,
    )
    save_table(pca_sanity_df, "ae_pca_sanity_check")

    # Top-20 FP vs Top-20 TP error analysis
    top20_detail_df, top20_dom_df = top20_fp_vs_tp_analysis(
        ae_model, X_val, y_val, ae_median["val_scores"],
        splits.feature_names, device=device,
    )
    save_table(top20_detail_df, "ae_top20_fp_vs_tp")
    save_table(top20_dom_df, "ae_top20_dominating_features")

    # Per-feature error analysis
    ae_model.eval()
    val_feat_err = compute_per_feature_error(ae_model, X_val, device=device)
    legit_feat_err = val_feat_err[y_val == 0].mean(axis=0)
    fraud_feat_err = val_feat_err[y_val == 1].mean(axis=0)

    feat_names = splits.feature_names
    feat_err_df = pd.DataFrame({
        "feature": feat_names,
        "legit_mean_sqerr": legit_feat_err,
        "fraud_mean_sqerr": fraud_feat_err,
        "fraud_over_legit_ratio": fraud_feat_err / (legit_feat_err + 1e-10),
    }).sort_values("fraud_over_legit_ratio", ascending=False)
    save_table(feat_err_df, "ae_per_feature_error")

    # Feature error bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = range(len(feat_names))
    ax.bar(x_pos, legit_feat_err, alpha=0.6, label="Legit", color="#2196F3")
    ax.bar(x_pos, fraud_feat_err, alpha=0.6, label="Fraud", color="#F44336")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feat_names, rotation=90, fontsize=7)
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Per-Feature Reconstruction Error: Legit vs Fraud")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "ae_per_feature_error")
    plt.close(fig)

    # ── 8. Engineered feature-set comparison (P2_03 requirement) ──
    _logger.info("=" * 70)
    _logger.info("ENGINEERED FEATURE SET COMPARISON")
    _logger.info("=" * 70)

    eng_splits = get_features("engineered")
    X_eng_train_legit = eng_splits.X_train_legit
    X_eng_val = eng_splits.X_val
    y_eng_val = eng_splits.y_val
    X_eng_test = eng_splits.X_test
    y_eng_test = eng_splits.y_test
    X_eng_val_legit = X_eng_val[y_eng_val == 0]
    d_eng = X_eng_train_legit.shape[1]

    _enable_determinism(ae_median["seed"])
    ae_eng_kwargs = {**ae_kwargs, "d_in": d_eng}
    ae_eng = UndercompleteAE(**ae_eng_kwargs)
    train_autoencoder(
        ae_eng, X_eng_train_legit, X_eng_val_legit,
        lr=1e-3, weight_decay=best_ae_wd, batch_size=512,
        max_epochs=100, patience=10, device=device, is_dae=False, verbose=False,
    )
    ae_eng_val_s = compute_reconstruction_mse(ae_eng, X_eng_val, device=device)
    ae_eng_test_s = compute_reconstruction_mse(ae_eng, X_eng_test, device=device)
    ae_eng_metrics = evaluate_scores(y_eng_val, ae_eng_val_s, y_eng_test, ae_eng_test_s, CFG)
    save_table(ae_eng_metrics, "ae_engineered_metrics")

    _enable_determinism(dae_median["seed"])
    dae_eng_kwargs = {**dae_kwargs, "d_in": d_eng}
    dae_eng = DenoisingAE(**dae_eng_kwargs)
    train_autoencoder(
        dae_eng, X_eng_train_legit, X_eng_val_legit,
        lr=1e-3, weight_decay=best_dae_wd, batch_size=512,
        max_epochs=100, patience=10, device=device, is_dae=True, verbose=False,
    )
    dae_eng_val_s = compute_reconstruction_mse(dae_eng, X_eng_val, device=device)
    dae_eng_test_s = compute_reconstruction_mse(dae_eng, X_eng_test, device=device)
    dae_eng_metrics = evaluate_scores(y_eng_val, dae_eng_val_s, y_eng_test, dae_eng_test_s, CFG)
    save_table(dae_eng_metrics, "ae_dae_engineered_metrics")

    base_vs_eng_df = pd.DataFrame([
        {
            "model": "AE (base)",
            "d_in": d_in,
            "val_pr_auc": round(pr_auc(y_val, ae_median["val_scores"]), 5),
            "test_pr_auc": round(pr_auc(y_test, ae_median["test_scores"]), 5),
            "val_recall@0.5%": round(recall_at_fpr(y_val, ae_median["val_scores"], 0.005), 5),
            "test_recall@0.5%": round(recall_at_fpr(y_test, ae_median["test_scores"], 0.005), 5),
        },
        {
            "model": "AE (engineered)",
            "d_in": d_eng,
            "val_pr_auc": round(pr_auc(y_eng_val, ae_eng_val_s), 5),
            "test_pr_auc": round(pr_auc(y_eng_test, ae_eng_test_s), 5),
            "val_recall@0.5%": round(recall_at_fpr(y_eng_val, ae_eng_val_s, 0.005), 5),
            "test_recall@0.5%": round(recall_at_fpr(y_eng_test, ae_eng_test_s, 0.005), 5),
        },
        {
            "model": "DAE (base)",
            "d_in": d_in,
            "val_pr_auc": round(pr_auc(y_val, dae_median["val_scores"]), 5),
            "test_pr_auc": round(pr_auc(y_test, dae_median["test_scores"]), 5),
            "val_recall@0.5%": round(recall_at_fpr(y_val, dae_median["val_scores"], 0.005), 5),
            "test_recall@0.5%": round(recall_at_fpr(y_test, dae_median["test_scores"], 0.005), 5),
        },
        {
            "model": "DAE (engineered)",
            "d_in": d_eng,
            "val_pr_auc": round(pr_auc(y_eng_val, dae_eng_val_s), 5),
            "test_pr_auc": round(pr_auc(y_eng_test, dae_eng_test_s), 5),
            "val_recall@0.5%": round(recall_at_fpr(y_eng_val, dae_eng_val_s, 0.005), 5),
            "test_recall@0.5%": round(recall_at_fpr(y_eng_test, dae_eng_test_s, 0.005), 5),
        },
    ])
    save_table(base_vs_eng_df, "ae_base_vs_engineered")

    # ── 9. Timing, technical validation, and parameter count ──
    _logger.info("=" * 70)
    _logger.info("TIMING & TECHNICAL VALIDATION")
    _logger.info("=" * 70)

    # Held-out legit batch vs fraud batch reconstruction error check
    legit_batch_mse = float(compute_reconstruction_mse(ae_model, X_val_legit[:512], device=device).mean())
    fraud_batch_mse = float(compute_reconstruction_mse(ae_model, X_val[y_val == 1], device=device).mean())
    dae_legit_batch_mse = float(compute_reconstruction_mse(dae_model, X_val_legit[:512], device=device).mean())
    dae_fraud_batch_mse = float(compute_reconstruction_mse(dae_model, X_val[y_val == 1], device=device).mean())

    # Inference timing per 10k transactions
    t0 = time.time()
    _ = compute_reconstruction_mse(ae_model, X_test[:10000], device=device)
    ae_inf_10k = time.time() - t0

    t0 = time.time()
    _ = compute_reconstruction_mse(dae_model, X_test[:10000], device=device)
    dae_inf_10k = time.time() - t0

    tech_val_df = pd.DataFrame([
        {
            "model": "UndercompleteAE",
            "n_params": ae_model.count_parameters(),
            "nan_detected": final_history["nan_detected"],
            "grad_norm_min": round(min(final_history["grad_norms"]), 4),
            "grad_norm_max": round(max(final_history["grad_norms"]), 4),
            "grad_norm_final": round(final_history["grad_norms"][-1], 4),
            "heldout_legit_batch_mse": round(legit_batch_mse, 5),
            "heldout_fraud_batch_mse": round(fraud_batch_mse, 5),
            "fraud_to_legit_mse_ratio": round(fraud_batch_mse / (legit_batch_mse + 1e-12), 2),
            "avg_epoch_sec": final_history["avg_epoch_time_seconds"],
            "total_train_sec": final_history["train_time_seconds"],
            "inference_10k_sec": round(ae_inf_10k, 4),
        },
        {
            "model": "DenoisingAE",
            "n_params": dae_model.count_parameters(),
            "nan_detected": final_dae_history["nan_detected"],
            "grad_norm_min": round(min(final_dae_history["grad_norms"]), 4),
            "grad_norm_max": round(max(final_dae_history["grad_norms"]), 4),
            "grad_norm_final": round(final_dae_history["grad_norms"][-1], 4),
            "heldout_legit_batch_mse": round(dae_legit_batch_mse, 5),
            "heldout_fraud_batch_mse": round(dae_fraud_batch_mse, 5),
            "fraud_to_legit_mse_ratio": round(dae_fraud_batch_mse / (dae_legit_batch_mse + 1e-12), 2),
            "avg_epoch_sec": final_dae_history["avg_epoch_time_seconds"],
            "total_train_sec": final_dae_history["train_time_seconds"],
            "inference_10k_sec": round(dae_inf_10k, 4),
        },
    ])
    save_table(tech_val_df, "ae_technical_validation")

    timing_data = {
        "ae_n_params": ae_model.count_parameters(),
        "dae_n_params": dae_model.count_parameters(),
        "ae_config": ae_model.get_config(),
        "dae_config": dae_model.get_config(),
        "inference_time_10k_sec": round(ae_inf_10k, 4),
        "dae_inference_time_10k_sec": round(dae_inf_10k, 4),
        "device": device,
    }
    timing_df = pd.DataFrame([{
        "model": "AE",
        "n_params": ae_model.count_parameters(),
        "architecture": str(ae_kwargs),
        "avg_epoch_sec": final_history["avg_epoch_time_seconds"],
        "total_train_sec": final_history["train_time_seconds"],
        "inference_10k_sec": round(ae_inf_10k, 4),
    }, {
        "model": "DAE",
        "n_params": dae_model.count_parameters(),
        "architecture": str(dae_kwargs),
        "avg_epoch_sec": final_dae_history["avg_epoch_time_seconds"],
        "total_train_sec": final_dae_history["train_time_seconds"],
        "inference_10k_sec": round(dae_inf_10k, 4),
    }])
    save_table(timing_df, "ae_timing")

    with open(_MODELS_DIR / "ae_timing.json", "w") as f:
        json.dump(timing_data, f, indent=2, default=str)

    _logger.info("=" * 70)
    _logger.info("PHASE 2 AE PIPELINE COMPLETE")
    _logger.info("=" * 70)
    _logger.info("AE params: %d  |  DAE params: %d", ae_model.count_parameters(), dae_model.count_parameters())
    _logger.info("Inference (10k samples): AE=%.4fs, DAE=%.4fs on %s", ae_inf_10k, dae_inf_10k, device)

    return {
        "ae_model": ae_model,
        "dae_model": dae_model,
        "ae_metrics": ae_metrics,
        "dae_metrics": dae_metrics,
        "hp_search": hp_df,
        "ae_seed_results": ae_seed_df,
        "dae_seed_results": dae_seed_df,
        "regularization": reg_df,
        "pca_sanity": pca_sanity_df,
        "base_vs_engineered": base_vs_eng_df,
        "technical_validation": tech_val_df,
    }


if __name__ == "__main__":
    run_phase2_ae()

