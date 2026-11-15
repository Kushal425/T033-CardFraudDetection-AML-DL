"""
src/hybrid_dagmm_lite.py
========================
Phase 2 Hybrid H3: DAGMM-lite (Decoupled Deep Autoencoding Gaussian Mixture Model).
Team T033 — Project 26: Card Fraud Triage.

Author: Prachee Dhar (Task P2_04)

Theoretical Motivation & Architecture:
--------------------------------------
1. Rationale (Zong et al., ICLR 2018 [VERIFY]):
   Standard density estimators in input space suffer from the curse of dimensionality.
   Standard autoencoders reconstruct data, but their scalar MSE reconstruction score
   discards orientation and manifold density information.
   DAGMM addresses this by feeding both the low-dimensional latent representation z
   and multi-aspect reconstruction error metrics into a Gaussian Mixture Model.

2. Architecture: Decoupled "DAGMM-lite":
   - Unlike full end-to-end DAGMM (which trains encoder, decoder, and estimation network
     jointly via a multi-task reconstruction + energy loss), our architecture is a
     modular, two-stage decoupled implementation:
       Stage 1: A trained Deep Autoencoder (Vani Rudra, P2_03) produces the bottleneck
                representation z = f(x) and decoded reconstruction x_hat = g(z).
       Stage 2: Reconstruction error features are explicitly derived:
                (i)   MSE: e_mse = mean((x - x_hat)^2)
                (ii)  Relative Euclidean distance: e_rel = ||x - x_hat||_2 / (||x||_2 + eps)
                (iii) Cosine distance: e_cos = 1 - (x . x_hat) / (||x||_2 ||x_hat||_2 + eps)
       Stage 3: The augmented representation v = [z, e_mse, e_rel, e_cos] is formed.
       Stage 4: A Gaussian Mixture Model is fit on v strictly using legitimate training
                samples (Class == 0).
       Stage 5: Anomaly score S(x) = -ln p(v) (negative log-likelihood).

3. Leakage Controls:
   - GMM is fitted only on legitimate training records (Class == 0).
   - Component count K and covariance structure are selected strictly on validation PR-AUC.
   - Test data is evaluated once with frozen parameters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.mixture import GaussianMixture

from src.autoencoder import UndercompleteAE, encode_dataset, load_autoencoder
from src.config import CFG, get_repo_root
from src.data import Splits, get_features
from src.evaluation import bootstrap_ci, paired_bootstrap_diff
from src.metrics import evaluate_scores, pr_auc, precision_at_k, recall_at_fpr, recall_at_k, roc_auc
from src.utils import get_logger, load_scores, save_fig, save_scores, save_table, set_seed

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruction Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reconstruction_features(
    model: UndercompleteAE,
    X: np.ndarray,
    batch_size: int = 2048,
    device: str = "cpu",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute latent representations z and multi-aspect reconstruction error features.

    Parameters
    ----------
    model : UndercompleteAE
        Trained autoencoder model in eval() mode.
    X : np.ndarray, shape (n_samples, d_in)
        Input features.
    batch_size : int
        Inference mini-batch size.
    device : str
        Target device ('cpu', 'cuda', etc.).
    eps : float
        Numerical stability constant.

    Returns
    -------
    Z : np.ndarray, shape (n_samples, d_z)
        Latent bottleneck codes.
    err_features : np.ndarray, shape (n_samples, 3)
        Columns: [e_mse, e_rel, e_cos]
    """
    model.eval()
    model = model.to(device)
    n = len(X)

    z_list = []
    err_list = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            x_b = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            z_b = model.encode(x_b)
            x_hat_b = model.decode(z_b)

            # 1. MSE
            mse = torch.mean((x_b - x_hat_b) ** 2, dim=1, keepdim=True)

            # 2. Relative Euclidean distance
            diff_norm = torch.norm(x_b - x_hat_b, p=2, dim=1, keepdim=True)
            x_norm = torch.norm(x_b, p=2, dim=1, keepdim=True)
            rel_err = diff_norm / (x_norm + eps)

            # 3. Cosine distance
            dot_prod = torch.sum(x_b * x_hat_b, dim=1, keepdim=True)
            x_hat_norm = torch.norm(x_hat_b, p=2, dim=1, keepdim=True)
            cos_sim = dot_prod / ((x_norm * x_hat_norm) + eps)
            cos_dist = 1.0 - cos_sim

            err_b = torch.cat([mse, rel_err, cos_dist], dim=1)

            z_list.append(z_b.cpu().numpy())
            err_list.append(err_b.cpu().numpy())

    Z = np.concatenate(z_list, axis=0).astype(np.float32)
    err_features = np.concatenate(err_list, axis=0).astype(np.float32)
    return Z, err_features


def build_dagmm_representation(
    Z: np.ndarray,
    err_features: np.ndarray,
    scaler_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Assemble and standardize the augmented DAGMM vector v = [z, err_features].

    Standardizes each feature dimension using mean and standard deviation from
    reference training data to prevent high-scale error metrics from dominating z.
    """
    v_raw = np.concatenate([Z, err_features], axis=1).astype(np.float32)
    if scaler_stats is None:
        mu = np.mean(v_raw, axis=0)
        sigma = np.std(v_raw, axis=0)
        sigma[sigma < 1e-12] = 1e-12
        scaler_stats = (mu, sigma)
    else:
        mu, sigma = scaler_stats

    v_norm = (v_raw - mu) / sigma
    return v_norm.astype(np.float32), scaler_stats


# ═══════════════════════════════════════════════════════════════════════════════
# GMM Fitting and Scoring for DAGMM-lite
# ═══════════════════════════════════════════════════════════════════════════════

def fit_dagmm_gmm(
    V_train_legit: np.ndarray,
    k: int = 4,
    cov_type: str = "diag",
    reg_covar: float = 1e-2,
    seed: int = 42,
) -> GaussianMixture:
    """
    Fit GMM on augmented legitimate training representations.
    """
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        reg_covar=reg_covar,
        random_state=seed,
        max_iter=150,
        n_init=2,
    )
    gmm.fit(V_train_legit)
    return gmm


def compute_dagmm_nll(gmm: GaussianMixture, V: np.ndarray) -> np.ndarray:
    """
    Compute negative log-likelihood anomaly score: S(v) = -score_samples(v).
    Higher score == lower probability under normal distribution == more anomalous.
    """
    log_prob = gmm.score_samples(V)
    scores = -log_prob
    return scores.astype(np.float32)


def run_dagmm_grid_search(
    V_train_legit: np.ndarray,
    V_val: np.ndarray,
    y_val: np.ndarray,
    V_test: np.ndarray,
    y_test: np.ndarray,
    k_candidates: Sequence[int] = (2, 4, 8, 12, 16),
    cov_types: Sequence[str] = ("diag", "tied", "full"),
    reg_covars: Sequence[float] = (1e-3, 1e-2, 1e-1),
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], GaussianMixture]:
    """
    Grid search over GMM hyperparameters for DAGMM-lite, selecting strictly on Val PR-AUC.
    """
    records = []
    best_val_pr = -1.0
    best_cfg: Dict[str, Any] = {}
    best_gmm: Optional[GaussianMixture] = None

    logger.info("Running DAGMM-lite GMM Grid Search...")
    for cov in cov_types:
        for k in k_candidates:
            for reg in reg_covars:
                try:
                    gmm = fit_dagmm_gmm(V_train_legit, k=k, cov_type=cov, reg_covar=reg, seed=seed)
                    s_val = compute_dagmm_nll(gmm, V_val)
                    s_test = compute_dagmm_nll(gmm, V_test)

                    val_pr = pr_auc(y_val, s_val)
                    val_roc = roc_auc(y_val, s_val)
                    val_rec05 = recall_at_fpr(y_val, s_val, fpr=0.005)

                    test_pr = pr_auc(y_test, s_test)
                    test_roc = roc_auc(y_test, s_test)
                    test_rec05 = recall_at_fpr(y_test, s_test, fpr=0.005)

                    records.append({
                        "k": k,
                        "covariance_type": cov,
                        "reg_covar": reg,
                        "converged": bool(gmm.converged_),
                        "bic_train": float(gmm.bic(V_train_legit[:5000])),  # sub-sample for speed
                        "val_pr_auc": val_pr,
                        "val_roc_auc": val_roc,
                        "val_recall_fpr0.5": val_rec05,
                        "test_pr_auc": test_pr,
                        "test_roc_auc": test_roc,
                        "test_recall_fpr0.5": test_rec05,
                    })

                    if val_pr > best_val_pr:
                        best_val_pr = val_pr
                        best_cfg = {
                            "k": k,
                            "covariance_type": cov,
                            "reg_covar": reg,
                            "val_pr_auc": val_pr,
                            "test_pr_auc": test_pr,
                        }
                        best_gmm = gmm
                except Exception as exc:
                    logger.warning("DAGMM grid fit failed for K=%d, cov=%s, reg=%.1e: %s", k, cov, reg, exc)

    df_grid = pd.DataFrame(records)
    logger.info("DAGMM grid complete. Selected on Val: K=%d, cov=%s, reg=%.1e -> Val PR-AUC=%.4f (Test PR-AUC=%.4f)",
                best_cfg["k"], best_cfg["covariance_type"], best_cfg["reg_covar"],
                best_cfg["val_pr_auc"], best_cfg["test_pr_auc"])
    return df_grid, best_cfg, best_gmm


# ═══════════════════════════════════════════════════════════════════════════════
# Alert Overlap and Complementarity Analytics
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_topk_alert_overlap(
    score_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    top_k_list: Sequence[int] = (50, 100, 200),
) -> pd.DataFrame:
    """
    Compute alert overlap, caught frauds, unique catches, and missed frauds across models.

    Parameters
    ----------
    score_dict : Dict[str, np.ndarray]
        Mapping model_name -> 1-D anomaly scores on the split.
    y : np.ndarray
        Ground-truth labels (1 = fraud, 0 = legit).
    top_k_list : Sequence[int]
        Queue capacities to inspect.

    Returns
    -------
    pd.DataFrame of overlap statistics.
    """
    fraud_indices = set(np.where(y == 1)[0])
    n_fraud = len(fraud_indices)
    records = []

    for k in top_k_list:
        model_topk: Dict[str, set] = {}
        model_frauds: Dict[str, set] = {}

        for name, s in score_dict.items():
            topk_idx = set(np.argsort(s)[::-1][:k])
            model_topk[name] = topk_idx
            model_frauds[name] = topk_idx.intersection(fraud_indices)

        for name in score_dict.keys():
            caught = model_frauds[name]
            # Union of frauds caught by other models
            other_frauds = set().union(*[model_frauds[m] for m in score_dict.keys() if m != name])
            unique_catches = caught - other_frauds

            records.append({
                "capacity_k": k,
                "model": name,
                "alerts_fired": k,
                "frauds_caught": len(caught),
                "precision_at_k": len(caught) / k,
                "recall_at_k": len(caught) / n_fraud,
                "unique_frauds_caught": len(unique_catches),
            })

    return pd.DataFrame(records)


def compute_cross_model_frauds_venn(
    s_ae: np.ndarray,
    s_gmm: np.ndarray,
    s_hybrid: np.ndarray,
    y: np.ndarray,
    capacity: int = 100,
) -> Dict[str, Any]:
    """
    Compute 3-way Venn partition between Standalone AE, Standalone GMM, and Hybrid.
    """
    fraud_idx = set(np.where(y == 1)[0])
    top_ae = set(np.argsort(s_ae)[::-1][:capacity]).intersection(fraud_idx)
    top_gmm = set(np.argsort(s_gmm)[::-1][:capacity]).intersection(fraud_idx)
    top_hyb = set(np.argsort(s_hybrid)[::-1][:capacity]).intersection(fraud_idx)

    only_ae = top_ae - top_gmm - top_hyb
    only_gmm = top_gmm - top_ae - top_hyb
    only_hyb = top_hyb - top_ae - top_gmm

    ae_gmm = (top_ae & top_gmm) - top_hyb
    ae_hyb = (top_ae & top_hyb) - top_gmm
    gmm_hyb = (top_gmm & top_hyb) - top_ae
    all_three = top_ae & top_gmm & top_hyb

    union_all = top_ae | top_gmm | top_hyb
    missed_all = fraud_idx - union_all

    return {
        "capacity": capacity,
        "total_frauds": len(fraud_idx),
        "only_ae": len(only_ae),
        "only_gmm": len(only_gmm),
        "only_hybrid": len(only_hyb),
        "ae_and_gmm_only": len(ae_gmm),
        "ae_and_hybrid_only": len(ae_hyb),
        "gmm_and_hybrid_only": len(gmm_hyb),
        "all_three": len(all_three),
        "union_caught": len(union_all),
        "missed_by_all": len(missed_all),
    }


def plot_hybrid_overlap_analysis(
    df_overlap: pd.DataFrame,
    venn_stats: Dict[str, Any],
    out_name: str = "hybrid_overlap_venn",
) -> None:
    """
    Plot bar chart of precision at capacity and Venn partition representation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Precision at capacity across models
    df_sub = df_overlap[df_overlap["capacity_k"] == 100]
    bars = ax1.bar(df_sub["model"], df_sub["precision_at_k"], color="steelblue", edgecolor="black")
    ax1.set_ylabel("Precision @ Top-100 Alerts", fontsize=11)
    ax1.set_title("Operational Precision @ Top-100 Capacity", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.2%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # 2. Venn / Partition breakdown of Top-100 frauds
    labels = [
        f"All Three\n({venn_stats['all_three']})",
        f"AE + Hyb\n({venn_stats['ae_and_hybrid_only']})",
        f"GMM + Hyb\n({venn_stats['gmm_and_hybrid_only']})",
        f"AE + GMM\n({venn_stats['ae_and_gmm_only']})",
        f"Only Hyb\n({venn_stats['only_hybrid']})",
        f"Only AE\n({venn_stats['only_ae']})",
        f"Only GMM\n({venn_stats['only_gmm']})",
    ]
    counts = [
        venn_stats["all_three"],
        venn_stats["ae_and_hybrid_only"],
        venn_stats["gmm_and_hybrid_only"],
        venn_stats["ae_and_gmm_only"],
        venn_stats["only_hybrid"],
        venn_stats["only_ae"],
        venn_stats["only_gmm"],
    ]

    colors = ["#2b5c8f", "#418ab3", "#59b5d9", "#9ecae1", "#d95f02", "#7570b3", "#e7298a"]
    ax2.bar(labels, counts, color=colors, edgecolor="black")
    ax2.set_ylabel(f"Number of Frauds Caught (Total in Test = {venn_stats['total_frauds']})", fontsize=11)
    ax2.set_title(f"Intersection Partition of Top-{venn_stats['capacity']} Frauds Caught", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    save_fig(fig, out_name)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Master Execution Pipeline for H3 & Cross-Hybrid Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_hybrid_dagmm_pipeline(
    weights_path: str = "results/models/ae_best.pt",
    config_path: str = "results/models/ae_config.json",
    save_official_scores: bool = True,
) -> Dict[str, Any]:
    """
    Execute full DAGMM-lite pipeline:
    1. Load Autoencoder and extract latent + reconstruction features.
    2. Build augmented representation v = [z, err].
    3. Grid search over GMM components and covariance structures.
    4. Save official scores:
       - results/scores/hybrid_dagmmlite_val.npy
       - results/scores/hybrid_dagmmlite_test.npy
    5. Perform comprehensive multi-hybrid comparison & alert overlap analysis:
       Compare AE, DAE, GMM, H1 (Latent GMM), H2 (Score Fusion), H3 (DAGMM-lite).
    """
    set_seed(42)
    logger.info("Starting Hybrid H3 DAGMM-lite Pipeline...")

    # 1. Load data and autoencoder
    splits = get_features("base")
    y_val = splits.y_val
    y_test = splits.y_test

    ae_model = load_autoencoder(weights_path, config_path)
    logger.info("Loaded Autoencoder: latent_dim=%d, hidden=%s", ae_model.latent_dim, ae_model.hidden)

    # 2. Extract latent codes and reconstruction error metrics
    logger.info("Computing reconstruction features on Train-Legit, Val, and Test...")
    Z_train_legit, err_train_legit = compute_reconstruction_features(ae_model, splits.X_train_legit)
    Z_val, err_val = compute_reconstruction_features(ae_model, splits.X_val)
    Z_test, err_test = compute_reconstruction_features(ae_model, splits.X_test)

    # 3. Standardize combined representation using Train-Legit statistics
    V_train_legit, scaler = build_dagmm_representation(Z_train_legit, err_train_legit)
    V_val, _ = build_dagmm_representation(Z_val, err_val, scaler_stats=scaler)
    V_test, _ = build_dagmm_representation(Z_test, err_test, scaler_stats=scaler)

    logger.info("Augmented DAGMM representation shape: Train-Legit=%s, Val=%s, Test=%s",
                V_train_legit.shape, V_val.shape, V_test.shape)

    # 4. Grid search (reduced for efficiency; full=expensive on 170k samples, removed)
    df_grid, best_cfg, best_gmm = run_dagmm_grid_search(
        V_train_legit, V_val, y_val, V_test, y_test,
        k_candidates=(2, 4, 8, 16),
        cov_types=("diag", "tied"),
        reg_covars=(1e-2, 1e-1),
        seed=42,
    )
    save_table(df_grid, "hybrid_dagmmlite_grid")

    # Plot component curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df_diag = df_grid[(df_grid["covariance_type"] == "diag") & (df_grid["reg_covar"] == 1e-2)]
    if not df_diag.empty:
        ax.plot(df_diag["k"], df_diag["val_pr_auc"], "o--", color="teal", label="Val PR-AUC (diag)")
        ax.plot(df_diag["k"], df_diag["test_pr_auc"], "s-", color="darkorange", label="Test PR-AUC (diag)")
    ax.axvline(best_cfg["k"], color="crimson", ls=":", label=f"Selected K = {best_cfg['k']}")
    ax.set_xlabel("Number of Mixture Components K")
    ax.set_ylabel("PR-AUC")
    ax.set_title("DAGMM-lite: PR-AUC vs Component Count K")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "hybrid_dagmmlite_components")
    plt.close(fig)

    # 5. Compute official H3 scores under best configuration
    s_dagmm_val = compute_dagmm_nll(best_gmm, V_val)
    s_dagmm_test = compute_dagmm_nll(best_gmm, V_test)

    if save_official_scores:
        save_scores("hybrid_dagmmlite", "val", s_dagmm_val)
        save_scores("hybrid_dagmmlite", "test", s_dagmm_test)
        logger.info("Saved official H3 score files: results/scores/hybrid_dagmmlite_{val,test}.npy")

    # 6. Bootstrap CI for H3
    ci_pr = bootstrap_ci(pr_auc, y_test, s_dagmm_test, n=1000, seed=42)
    ci_rec05 = bootstrap_ci(recall_at_fpr, y_test, s_dagmm_test, fpr=0.005, n=1000, seed=42)

    logger.info("H3 Official Results: Test PR-AUC = %.4f (95%% CI: [%.4f, %.4f]) | Recall@0.5%% = %.4f (95%% CI: [%.4f, %.4f])",
                ci_pr["point"], ci_pr["ci_lo"], ci_pr["ci_hi"],
                ci_rec05["point"], ci_rec05["ci_lo"], ci_rec05["ci_hi"])

    # 7. Multi-Hybrid Comparative Evaluation (AE, DAE, GMM, H1, H2, H3)
    score_models = {
        "Standalone AE": (load_scores("ae", "val"), load_scores("ae", "test")),
        "Standalone DAE": (load_scores("ae_dae", "val"), load_scores("ae_dae", "test")),
        "Standalone GMM": (load_scores("gmm", "val"), load_scores("gmm", "test")),
        "H1: Latent GMM": (load_scores("hybrid_latentgmm", "val"), load_scores("hybrid_latentgmm", "test")),
        "H2: Score Fusion": (load_scores("hybrid_fusion", "val"), load_scores("hybrid_fusion", "test")),
        "H3: DAGMM-lite": (s_dagmm_val, s_dagmm_test),
    }

    comp_records = []
    test_scores_dict = {}

    for name, (s_v, s_t) in score_models.items():
        test_scores_dict[name] = s_t
        eval_df = evaluate_scores(y_val, s_v, y_test, s_t, cfg=CFG)
        ci = bootstrap_ci(pr_auc, y_test, s_t, n=1000, seed=42)

        comp_records.append({
            "model": name,
            "val_pr_auc": float(eval_df.loc["PR-AUC", "val"]),
            "val_roc_auc": float(eval_df.loc["ROC-AUC", "val"]),
            "val_recall_fpr0.1": float(eval_df.loc["recall@FPR=0.001", "val"]),
            "val_recall_fpr0.5": float(eval_df.loc["recall@FPR=0.005", "val"]),
            "test_pr_auc": float(eval_df.loc["PR-AUC", "test"]),
            "test_pr_auc_ci_lo": float(ci["ci_lo"]),
            "test_pr_auc_ci_hi": float(ci["ci_hi"]),
            "test_roc_auc": float(eval_df.loc["ROC-AUC", "test"]),
            "test_recall_fpr0.1": float(eval_df.loc["recall@FPR=0.001", "test"]),
            "test_recall_fpr0.5": float(eval_df.loc["recall@FPR=0.005", "test"]),
            "test_recall_fpr1.0": float(eval_df.loc["recall@FPR=0.010", "test"]) if "recall@FPR=0.010" in eval_df.index else np.nan,
            "test_prec_50": float(eval_df.loc["precision@K=50", "test"]),
            "test_prec_100": float(eval_df.loc["precision@K=100", "test"]),
            "test_prec_200": float(eval_df.loc["precision@K=200", "test"]),
        })

    df_comp = pd.DataFrame(comp_records)
    save_table(df_comp, "hybrid_comparison")

    # 8. Paired bootstrap significance tests against Standalone AE and GMM
    s_ae_test = score_models["Standalone AE"][1]
    s_gmm_test = score_models["Standalone GMM"][1]
    paired_records = []

    for name in ["H1: Latent GMM", "H2: Score Fusion", "H3: DAGMM-lite"]:
        s_h = score_models[name][1]
        diff_ae = paired_bootstrap_diff(y_test, s_h, s_ae_test, metric_fn=pr_auc, n=1000, seed=42)
        diff_gmm = paired_bootstrap_diff(y_test, s_h, s_gmm_test, metric_fn=pr_auc, n=1000, seed=42)

        paired_records.append({
            "hybrid_model": name,
            "delta_pr_vs_ae": diff_ae["diff_point"],
            "ci_lo_vs_ae": diff_ae["ci_lo"],
            "ci_hi_vs_ae": diff_ae["ci_hi"],
            "significant_vs_ae": diff_ae["significant"],
            "delta_pr_vs_gmm": diff_gmm["diff_point"],
            "ci_lo_vs_gmm": diff_gmm["ci_lo"],
            "ci_hi_vs_gmm": diff_gmm["ci_hi"],
            "significant_vs_gmm": diff_gmm["significant"],
        })

    df_paired = pd.DataFrame(paired_records)
    save_table(df_paired, "hybrid_paired_bootstrap")

    # 9. Top-K Alert Overlap and Venn Analysis
    df_overlap = analyze_topk_alert_overlap(test_scores_dict, y_test, top_k_list=(50, 100, 200))
    save_table(df_overlap, "hybrid_overlap")

    venn_stats = compute_cross_model_frauds_venn(
        s_ae=s_ae_test,
        s_gmm=s_gmm_test,
        s_hybrid=score_models["H2: Score Fusion"][1],
        y=y_test,
        capacity=100,
    )
    df_venn = pd.DataFrame([venn_stats])
    save_table(df_venn, "hybrid_overlap_venn_breakdown")
    plot_hybrid_overlap_analysis(df_overlap, venn_stats, out_name="hybrid_overlap_venn")

    # 10. Publication PR curves comparing Hybrids vs Parents
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, s_t in test_scores_dict.items():
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(y_test, s_t)
        pr_val = pr_auc(y_test, s_t)
        ax.plot(rec, prec, lw=1.8, label=f"{name} (PR-AUC={pr_val:.4f})")

    ax.axhline(float(y_test.mean()), color="gray", ls=":", label=f"Base Rate ({float(y_test.mean()):.4%})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves: Phase 2 Hybrids vs Baseline Parents (Test Set)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_fig(fig, "hybrid_pr_curves")
    plt.close(fig)

    return {
        "best_cfg": best_cfg,
        "ci_pr_auc": [float(ci_pr["ci_lo"]), float(ci_pr["ci_hi"])],
        "df_comp": df_comp,
        "df_paired": df_paired,
        "venn_stats": venn_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid H3 DAGMM-lite & Overlap Analytics")
    parser.add_argument("--weights", type=str, default="results/models/ae_best.pt")
    parser.add_argument("--config", type=str, default="results/models/ae_config.json")
    args = parser.parse_args()

    run_hybrid_dagmm_pipeline(weights_path=args.weights, config_path=args.config)

