"""
src/hybrid_fusion.py
====================
Phase 2 Hybrid H2: Anomaly Score Fusion for Card Fraud Triage.
Team T033 — Project 26: Card Fraud Triage.

Author: Prachee Dhar (Task P2_04)

Theoretical and Operational Motivation:
---------------------------------------
1. Rationale:
   - Deep Autoencoders (AE) detect anomalies by measuring reconstruction fidelity:
     samples that deviate from the normal manifold incur high MSE. However, AEs can
     over-generalise (reconstructing novel fraud patterns with moderate error) and
     do not estimate probability density.
   - Gaussian Mixture Models (GMM) provide an explicit probabilistic density model
     p(x) (or p(z) in latent space), yielding calibrated negative log-likelihood (NLL)
     scores. However, GMMs suffer from the curse of dimensionality and covariance
     ill-conditioning in high-dimensional feature spaces.
   - Score fusion (Hybrid H2) combines these complementary perspectives:
     reconstruction error (distance to the manifold) + density estimate (sparsity
     along the manifold).

2. Fusion Formulations:
   - Normalized Weighted Fusion:
       S_hybrid = alpha * S_AE_norm + (1 - alpha) * S_GMM_norm
     where alpha in [0, 1] is chosen strictly on the validation set.
   - Max Fusion:
       S_max = max(S_AE_norm, S_GMM_norm)
     capturing anomalies that deviate strongly along either dimension.
   - Rank-Average Fusion:
       S_rank = (rank(S_AE) + rank(S_GMM)) / (2 * N)
     invariant to score distributions and monotonic non-linearities.

3. Normalization Schemes Evaluated:
   - Standardisation (z-score) using legitimate training statistics.
   - Log-standardisation: log(S + eps) followed by z-score (handles heavy tails).
   - Min-Max scaling to [0, 1].
   - Cumulative distribution / Rank transformation.

4. Leakage Prevention:
   - All normalisation statistics and hyperparameter selections (alpha) are fit
     or selected strictly on training/validation data. Test scores are computed
     once under frozen parameters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from scipy import stats

from src.config import CFG, get_repo_root
from src.data import Splits, get_features
from src.evaluation import bootstrap_ci, paired_bootstrap_diff
from src.metrics import evaluate_scores, pr_auc, recall_at_fpr, roc_auc
from src.utils import get_logger, load_scores, save_fig, save_scores, save_table, set_seed

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_zscore(
    scores: np.ndarray,
    ref_scores: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """
    Standardize scores using reference (train-legit) mean and standard deviation.

    Parameters
    ----------
    scores : np.ndarray
        Array of anomaly scores to normalize.
    ref_scores : np.ndarray, optional
        Reference scores (e.g. from training legitimate samples) used to compute
        mean and std. If None, computes statistics on `scores`.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    normalized_scores : np.ndarray
    mean : float
    std : float
    """
    ref = ref_scores if ref_scores is not None else scores
    mu = float(np.mean(ref))
    sigma = float(np.std(ref))
    if sigma < eps:
        sigma = eps
    normed = (scores - mu) / sigma
    return normed.astype(np.float32), mu, sigma


def normalize_log_zscore(
    scores: np.ndarray,
    ref_scores: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float, float]:
    """
    Apply log transform followed by standardization.
    Effective for heavy-tailed score distributions like reconstruction MSE and NLL.
    """
    min_val = float(np.min(ref_scores if ref_scores is not None else scores))
    shift = max(0.0, -min_val) + eps
    
    log_ref = np.log((ref_scores if ref_scores is not None else scores) + shift)
    mu = float(np.mean(log_ref))
    sigma = float(np.std(log_ref))
    if sigma < 1e-12:
        sigma = 1e-12
        
    log_scores = np.log(scores + shift)
    normed = (log_scores - mu) / sigma
    return normed.astype(np.float32), mu, sigma


def normalize_minmax(
    scores: np.ndarray,
    ref_scores: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """
    Min-max scale scores to [0, 1] based on reference minimum and maximum.
    """
    ref = ref_scores if ref_scores is not None else scores
    s_min = float(np.min(ref))
    s_max = float(np.max(ref))
    denom = max(s_max - s_min, eps)
    normed = (scores - s_min) / denom
    return normed.astype(np.float32), s_min, s_max


def normalize_rank(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to fractional ranks in [0, 1].
    Higher raw score -> higher rank (more anomalous).
    """
    n = len(scores)
    if n <= 1:
        return np.zeros_like(scores, dtype=np.float32)
    # scipy.stats.rankdata gives ranks 1 .. n
    ranks = stats.rankdata(scores, method="average")
    normed = (ranks - 1.0) / (n - 1.0)
    return normed.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Fusion Operators
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_weighted(
    score_ae: np.ndarray,
    score_gmm: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Normalized weighted fusion:
        S = alpha * S_ae + (1 - alpha) * S_gmm
    """
    return (alpha * score_ae + (1.0 - alpha) * score_gmm).astype(np.float32)


def fuse_max(score_ae: np.ndarray, score_gmm: np.ndarray) -> np.ndarray:
    """
    Max fusion:
        S = max(S_ae, S_gmm)
    """
    return np.maximum(score_ae, score_gmm).astype(np.float32)


def fuse_rank_average(score_ae: np.ndarray, score_gmm: np.ndarray) -> np.ndarray:
    """
    Rank-average fusion:
        S = 0.5 * (rank(S_ae) + rank(S_gmm))
    """
    r_ae = normalize_rank(score_ae)
    r_gmm = normalize_rank(score_gmm)
    return (0.5 * (r_ae + r_gmm)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Experimental Workflows
# ═══════════════════════════════════════════════════════════════════════════════

def run_score_correlation_analysis(
    s_ae_val: np.ndarray,
    s_gmm_val: np.ndarray,
    s_ae_test: np.ndarray,
    s_gmm_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    save_plot: bool = True,
) -> Dict[str, float]:
    """
    Analyze Pearson and Spearman rank correlation between Autoencoder and GMM scores.
    """
    spearman_val, _ = stats.spearmanr(s_ae_val, s_gmm_val)
    pearson_val, _ = stats.pearsonr(s_ae_val, s_gmm_val)
    spearman_test, _ = stats.spearmanr(s_ae_test, s_gmm_test)
    pearson_test, _ = stats.pearsonr(s_ae_test, s_gmm_test)

    # Class-stratified correlations on test
    legit_mask = (y_test == 0)
    fraud_mask = (y_test == 1)
    spearman_legit, _ = stats.spearmanr(s_ae_test[legit_mask], s_gmm_test[legit_mask])
    spearman_fraud, _ = stats.spearmanr(s_ae_test[fraud_mask], s_gmm_test[fraud_mask])

    res = {
        "spearman_val": float(spearman_val),
        "pearson_val": float(pearson_val),
        "spearman_test": float(spearman_test),
        "pearson_test": float(pearson_test),
        "spearman_test_legit": float(spearman_legit),
        "spearman_test_fraud": float(spearman_fraud),
    }

    logger.info(
        "Score Correlation — Val Spearman: %.4f (Pearson: %.4f) | "
        "Test Spearman: %.4f (Pearson: %.4f) | Test Fraud Spearman: %.4f",
        spearman_val, pearson_val, spearman_test, pearson_test, spearman_fraud,
    )

    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validation scatter (sample legitimate to avoid rendering tens of thousands of dots)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(np.where(legit_mask)[0], size=min(2000, legit_mask.sum()), replace=False)
        
        for ax, (s_ae, s_gmm, y, title) in zip(
            axes,
            [(s_ae_val, s_gmm_val, y_val, "Validation Set"), (s_ae_test, s_gmm_test, y_test, "Test Set")]
        ):
            l_idx = np.where(y == 0)[0]
            f_idx = np.where(y == 1)[0]
            sub_l = rng.choice(l_idx, size=min(2000, len(l_idx)), replace=False)
            
            ax.scatter(s_ae[sub_l], s_gmm[sub_l], alpha=0.3, s=15, c="steelblue", label="Legitimate (sample)")
            ax.scatter(s_ae[f_idx], s_gmm[f_idx], alpha=0.85, s=30, c="crimson", marker="^", label=f"Fraud (n={len(f_idx)})")
            ax.set_xlabel("AE Reconstruction Score (MSE)")
            ax.set_ylabel("GMM Anomaly Score (NLL)")
            ax.set_title(f"{title} Score Distribution")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left")

        plt.tight_layout()
        save_fig(fig, "hybrid_score_correlation")
        plt.close(fig)

    return res


def sweep_fusion_weights(
    s_ae_val: np.ndarray,
    s_gmm_val: np.ndarray,
    y_val: np.ndarray,
    s_ae_test: np.ndarray,
    s_gmm_test: np.ndarray,
    y_test: np.ndarray,
    norm_method: str = "zscore",
    weights: Optional[Sequence[float]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Sweep alpha in [0.0, 1.0] to find the optimal fusion weight on the validation set.
    Evaluates test set performance for transparency, but selection is STRICTLY on val.
    """
    if weights is None:
        weights = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]

    # Normalize validation and test scores using validation/training stats
    if norm_method == "zscore":
        ae_val_norm, mu_ae, sig_ae = normalize_zscore(s_ae_val)
        gmm_val_norm, mu_gmm, sig_gmm = normalize_zscore(s_gmm_val)
        ae_test_norm, _, _ = normalize_zscore(s_ae_test, ref_scores=s_ae_val)
        gmm_test_norm, _, _ = normalize_zscore(s_gmm_test, ref_scores=s_gmm_val)
    elif norm_method == "log_zscore":
        ae_val_norm, _, _ = normalize_log_zscore(s_ae_val)
        gmm_val_norm, _, _ = normalize_log_zscore(s_gmm_val)
        ae_test_norm, _, _ = normalize_log_zscore(s_ae_test, ref_scores=s_ae_val)
        gmm_test_norm, _, _ = normalize_log_zscore(s_gmm_test, ref_scores=s_gmm_val)
    elif norm_method == "minmax":
        ae_val_norm, _, _ = normalize_minmax(s_ae_val)
        gmm_val_norm, _, _ = normalize_minmax(s_gmm_val)
        ae_test_norm, _, _ = normalize_minmax(s_ae_test, ref_scores=s_ae_val)
        gmm_test_norm, _, _ = normalize_minmax(s_gmm_test, ref_scores=s_gmm_val)
    elif norm_method == "rank":
        ae_val_norm = normalize_rank(s_ae_val)
        gmm_val_norm = normalize_rank(s_gmm_val)
        ae_test_norm = normalize_rank(s_ae_test)
        gmm_test_norm = normalize_rank(s_gmm_test)
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")

    records = []
    best_val_pr = -1.0
    best_alpha = 0.5

    for w in weights:
        s_val_fused = fuse_weighted(ae_val_norm, gmm_val_norm, alpha=w)
        s_test_fused = fuse_weighted(ae_test_norm, gmm_test_norm, alpha=w)

        v_pr = pr_auc(y_val, s_val_fused)
        v_roc = roc_auc(y_val, s_val_fused)
        v_rec_01 = recall_at_fpr(y_val, s_val_fused, fpr=0.001)
        v_rec_05 = recall_at_fpr(y_val, s_val_fused, fpr=0.005)

        t_pr = pr_auc(y_test, s_test_fused)
        t_roc = roc_auc(y_test, s_test_fused)
        t_rec_01 = recall_at_fpr(y_test, s_test_fused, fpr=0.001)
        t_rec_05 = recall_at_fpr(y_test, s_test_fused, fpr=0.005)

        if v_pr > best_val_pr:
            best_val_pr = v_pr
            best_alpha = float(w)

        records.append({
            "norm_method": norm_method,
            "alpha_ae": float(w),
            "alpha_gmm": float(1.0 - w),
            "val_pr_auc": v_pr,
            "val_roc_auc": v_roc,
            "val_recall_fpr0.1": v_rec_01,
            "val_recall_fpr0.5": v_rec_05,
            "test_pr_auc": t_pr,
            "test_roc_auc": t_roc,
            "test_recall_fpr0.1": t_rec_01,
            "test_recall_fpr0.5": t_rec_05,
        })

    df = pd.DataFrame(records)
    logger.info("Weight sweep (%s): best alpha_ae=%.2f yields Val PR-AUC=%.4f (Test PR-AUC=%.4f)",
                norm_method, best_alpha, best_val_pr,
                df.loc[df["alpha_ae"] == best_alpha, "test_pr_auc"].values[0])
    return df, best_alpha


def compare_fusion_methods(
    s_ae_val: np.ndarray,
    s_gmm_val: np.ndarray,
    y_val: np.ndarray,
    s_ae_test: np.ndarray,
    s_gmm_test: np.ndarray,
    y_test: np.ndarray,
    best_alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Compare multiple fusion strategies against standalone parents on validation and test sets.
    """
    ae_val_z, _, _ = normalize_zscore(s_ae_val)
    gmm_val_z, _, _ = normalize_zscore(s_gmm_val)
    ae_test_z, _, _ = normalize_zscore(s_ae_test, ref_scores=s_ae_val)
    gmm_test_z, _, _ = normalize_zscore(s_gmm_test, ref_scores=s_gmm_val)

    methods = {
        "Standalone AE": (s_ae_val, s_ae_test),
        "Standalone GMM": (s_gmm_val, s_gmm_test),
        f"Weighted Fusion (alpha={best_alpha:.1f})": (
            fuse_weighted(ae_val_z, gmm_val_z, best_alpha),
            fuse_weighted(ae_test_z, gmm_test_z, best_alpha),
        ),
        "Max Fusion (z-score)": (
            fuse_max(ae_val_z, gmm_val_z),
            fuse_max(ae_test_z, gmm_test_z),
        ),
        "Rank Average Fusion": (
            fuse_rank_average(s_ae_val, s_gmm_val),
            fuse_rank_average(s_ae_test, s_gmm_test),
        ),
    }

    records = []
    for name, (s_v, s_t) in methods.items():
        df_eval = evaluate_scores(y_val, s_v, y_test, s_t, cfg=CFG)

        records.append({
            "method": name,
            "val_pr_auc": float(df_eval.loc["PR-AUC", "val"]),
            "val_roc_auc": float(df_eval.loc["ROC-AUC", "val"]),
            "val_recall_fpr0.1": float(df_eval.loc["recall@FPR=0.001", "val"]),
            "val_recall_fpr0.5": float(df_eval.loc["recall@FPR=0.005", "val"]),
            "test_pr_auc": float(df_eval.loc["PR-AUC", "test"]),
            "test_roc_auc": float(df_eval.loc["ROC-AUC", "test"]),
            "test_recall_fpr0.1": float(df_eval.loc["recall@FPR=0.001", "test"]),
            "test_recall_fpr0.5": float(df_eval.loc["recall@FPR=0.005", "test"]),
            "test_prec_50": float(df_eval.loc["precision@K=50", "test"]) if "precision@K=50" in df_eval.index else np.nan,
            "test_prec_100": float(df_eval.loc["precision@K=100", "test"]) if "precision@K=100" in df_eval.index else np.nan,
        })

    return pd.DataFrame(records)


def plot_fusion_weight_sensitivity(
    df_weights: pd.DataFrame,
    best_alpha: float,
    out_name: str = "hybrid_fusion_sensitivity",
) -> None:
    """
    Plot PR-AUC and Recall@0.5% FPR sensitivity across fusion weights alpha.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    alphas = df_weights["alpha_ae"].values
    val_pr = df_weights["val_pr_auc"].values
    test_pr = df_weights["test_pr_auc"].values

    ax1.plot(alphas, val_pr, "o--", color="teal", lw=2, label="Validation PR-AUC")
    ax1.plot(alphas, test_pr, "s-", color="darkorange", lw=2, label="Test PR-AUC (frozen)")
    ax1.axvline(best_alpha, color="crimson", ls=":", lw=1.5, label=f"Selected alpha = {best_alpha:.2f} (Val max)")
    ax1.set_xlabel(r"Autoencoder Weight $\alpha$ ($S = \alpha S_{\mathrm{AE}} + (1-\alpha) S_{\mathrm{GMM}}$)", fontsize=11)
    ax1.set_ylabel("PR-AUC", fontsize=11)
    ax1.set_title("PR-AUC vs Fusion Weight", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best")

    val_rec = df_weights["val_recall_fpr0.5"].values
    test_rec = df_weights["test_recall_fpr0.5"].values
    ax2.plot(alphas, val_rec, "o--", color="teal", lw=2, label="Validation Recall@0.5% FPR")
    ax2.plot(alphas, test_rec, "s-", color="darkorange", lw=2, label="Test Recall@0.5% FPR")
    ax2.axvline(best_alpha, color="crimson", ls=":", lw=1.5, label=f"Selected alpha = {best_alpha:.2f}")
    ax2.set_xlabel(r"Autoencoder Weight $\alpha$", fontsize=11)
    ax2.set_ylabel("Recall @ 0.5% FPR", fontsize=11)
    ax2.set_title("Operational Recall vs Fusion Weight", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best")

    plt.tight_layout()
    save_fig(fig, out_name)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_hybrid_fusion_pipeline(
    ae_name: str = "ae_dae",
    gmm_name: str = "gmm",
    save_official_scores: bool = True,
) -> Dict[str, Any]:
    """
    Execute full H2 score fusion pipeline:
    1. Load AE and GMM score arrays.
    2. Check score contracts.
    3. Analyze score correlation.
    4. Run normalization comparison & weight sensitivity.
    5. Select optimal alpha strictly on validation PR-AUC.
    6. Save official fused scores:
       - results/scores/hybrid_fusion_val.npy
       - results/scores/hybrid_fusion_test.npy
    7. Generate comparison tables and figures.
    """
    set_seed(42)
    logger.info("Starting Hybrid H2 Score Fusion Pipeline (%s + %s)...", ae_name, gmm_name)

    # 1. Load data labels and scores
    splits = get_features("base")
    y_val = splits.y_val
    y_test = splits.y_test

    s_ae_val = load_scores(ae_name, "val")
    s_ae_test = load_scores(ae_name, "test")
    s_gmm_val = load_scores(gmm_name, "val")
    s_gmm_test = load_scores(gmm_name, "test")

    # Contract checks
    assert len(s_ae_val) == len(y_val), f"AE val length mismatch: {len(s_ae_val)} vs {len(y_val)}"
    assert len(s_ae_test) == len(y_test), f"AE test length mismatch: {len(s_ae_test)} vs {len(y_test)}"
    assert len(s_gmm_val) == len(y_val), f"GMM val length mismatch: {len(s_gmm_val)} vs {len(y_val)}"
    assert len(s_gmm_test) == len(y_test), f"GMM test length mismatch: {len(s_gmm_test)} vs {len(y_test)}"
    assert not np.isnan(s_ae_val).any() and not np.isnan(s_gmm_val).any(), "NaN found in validation scores"

    # 2. Correlation analysis
    corr_results = run_score_correlation_analysis(
        s_ae_val, s_gmm_val, s_ae_test, s_gmm_test, y_val, y_test, save_plot=True
    )
    df_corr = pd.DataFrame([corr_results])
    save_table(df_corr, "hybrid_fusion_correlation")

    # 3. Sweep weights across normalizations
    df_zscore, best_alpha_z = sweep_fusion_weights(
        s_ae_val, s_gmm_val, y_val, s_ae_test, s_gmm_test, y_test, norm_method="zscore"
    )
    df_minmax, best_alpha_minmax = sweep_fusion_weights(
        s_ae_val, s_gmm_val, y_val, s_ae_test, s_gmm_test, y_test, norm_method="minmax"
    )
    df_logz, best_alpha_logz = sweep_fusion_weights(
        s_ae_val, s_gmm_val, y_val, s_ae_test, s_gmm_test, y_test, norm_method="log_zscore"
    )
    df_rank, best_alpha_rank = sweep_fusion_weights(
        s_ae_val, s_gmm_val, y_val, s_ae_test, s_gmm_test, y_test, norm_method="rank"
    )

    df_all_sweeps = pd.concat([df_zscore, df_minmax, df_logz, df_rank], ignore_index=True)
    save_table(df_all_sweeps, "hybrid_fusion_weights")

    # Plot sensitivity for primary normalization (z-score)
    plot_fusion_weight_sensitivity(df_zscore, best_alpha_z, out_name="hybrid_fusion_sensitivity")

    # 4. Compare fusion methods
    df_methods = compare_fusion_methods(
        s_ae_val, s_gmm_val, y_val, s_ae_test, s_gmm_test, y_test, best_alpha=best_alpha_z
    )
    save_table(df_methods, "hybrid_fusion_methods")

    # 5. Compute official selected scores
    ae_val_z, _, _ = normalize_zscore(s_ae_val)
    gmm_val_z, _, _ = normalize_zscore(s_gmm_val)
    ae_test_z, _, _ = normalize_zscore(s_ae_test, ref_scores=s_ae_val)
    gmm_test_z, _, _ = normalize_zscore(s_gmm_test, ref_scores=s_gmm_val)

    s_fused_val = fuse_weighted(ae_val_z, gmm_val_z, alpha=best_alpha_z)
    s_fused_test = fuse_weighted(ae_test_z, gmm_test_z, alpha=best_alpha_z)

    if save_official_scores:
        save_scores("hybrid_fusion", "val", s_fused_val)
        save_scores("hybrid_fusion", "test", s_fused_test)
        logger.info("Saved official H2 score files: results/scores/hybrid_fusion_{val,test}.npy")

    # 6. Bootstrap confidence interval for official H2
    ci_pr = bootstrap_ci(pr_auc, y_test, s_fused_test, n=1000, seed=42)
    ci_rec05 = bootstrap_ci(recall_at_fpr, y_test, s_fused_test, fpr=0.005, n=1000, seed=42)
    diff_vs_ae = paired_bootstrap_diff(y_test, s_fused_test, s_ae_test, metric_fn=pr_auc, n=1000, seed=42)
    diff_vs_gmm = paired_bootstrap_diff(y_test, s_fused_test, s_gmm_test, metric_fn=pr_auc, n=1000, seed=42)

    logger.info(
        "H2 Official Results (alpha=%.2f): Test PR-AUC = %.4f (95%% CI: [%.4f, %.4f]) | "
        "Recall@0.5%% FPR = %.4f (95%% CI: [%.4f, %.4f])",
        best_alpha_z, ci_pr["point"], ci_pr["ci_lo"], ci_pr["ci_hi"],
        ci_rec05["point"], ci_rec05["ci_lo"], ci_rec05["ci_hi"],
    )
    logger.info("Paired diff vs AE: Delta=%.4f (significant=%s) | vs GMM: Delta=%.4f (significant=%s)",
                diff_vs_ae["diff_point"], diff_vs_ae["significant"],
                diff_vs_gmm["diff_point"], diff_vs_gmm["significant"])

    return {
        "best_alpha": best_alpha_z,
        "val_pr_auc": float(df_zscore.loc[df_zscore["alpha_ae"] == best_alpha_z, "val_pr_auc"].values[0]),
        "test_pr_auc": float(ci_pr["point"]),
        "ci_pr_auc": [float(ci_pr["ci_lo"]), float(ci_pr["ci_hi"])],
        "test_recall_05": float(ci_rec05["point"]),
        "ci_recall_05": [float(ci_rec05["ci_lo"]), float(ci_rec05["ci_hi"])],
        "diff_vs_ae": diff_vs_ae,
        "diff_vs_gmm": diff_vs_gmm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid H2 Score Fusion")
    parser.add_argument("--ae", type=str, default="ae_dae", help="Autoencoder score prefix (ae or ae_dae)")
    parser.add_argument("--gmm", type=str, default="gmm", help="GMM score prefix (gmm or hybrid_latentgmm)")
    args = parser.parse_args()

    run_hybrid_fusion_pipeline(ae_name=args.ae, gmm_name=args.gmm)

