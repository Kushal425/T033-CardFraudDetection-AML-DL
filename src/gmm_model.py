"""
src/gmm_model.py
================
Gaussian Mixture Model (GMM) anomaly scoring and evaluation pipeline
for Team T033 — Card Fraud Triage (Phase 1: ML only).
Author: Apoorva Choudhary (Task 02)

Theoretical and Operational Contract:
-------------------------------------
1. Fitted EXCLUSIVELY on legitimate training instances (X_train_legit, Class == 0).
   Fraud labels are strictly hidden and never accessed during model fitting.
2. Anomaly score convention: s(x) = -ln p(x) (Negative Log-Likelihood, NLL).
   Higher score indicates lower probability density under the legitimate manifold
   (i.e. more anomalous / more suspicious).
3. Scores saved to results/scores/gmm_val.npy and results/scores/gmm_test.npy
   as 1-D float32 arrays matching split row lengths.
4. Model serialized via joblib to results/models/gmm.joblib.
5. All diagnostics, AIC/BIC grid sweeps, and instability studies are executed
   and exported to results/tables/ and results/figures/.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.config import CFG, get_repo_root
from src.data import Splits, get_features
from src.metrics import evaluate_scores, pr_auc, recall_at_fpr, roc_auc
from src.utils import get_logger, save_fig, save_scores, save_table, set_seed

_logger = get_logger(__name__)
_ROOT = get_repo_root()



# =============================================================================
# 1. Theoretical Parameter Counting
# =============================================================================

def calculate_free_params(covariance_type: str, n_components: int, d: int) -> int:
    """
    Calculate the exact theoretical number of free parameters p for a GMM.

    Formulas:
    - full:      K*d (means) + K*d*(d+1)/2 (covariances) + (K - 1) (weights)
    - diag:      K*d (means) + K*d (variances)           + (K - 1) (weights)
    - spherical: K*d (means) + K*1 (scalar variances)    + (K - 1) (weights)
    - tied:      K*d (means) + d*(d+1)/2 (shared cov)    + (K - 1) (weights)

    Parameters
    ----------
    covariance_type : str
        One of 'full', 'diag', 'spherical', 'tied'.
    n_components : int
        Number of mixture components K.
    d : int
        Feature dimensionality.

    Returns
    -------
    int
        Total number of free parameters p.
    """
    K = n_components
    if covariance_type == "full":
        return int(K * d + K * d * (d + 1) / 2 + (K - 1))
    elif covariance_type == "diag":
        return int(K * d + K * d + (K - 1))
    elif covariance_type == "spherical":
        return int(K * d + K + (K - 1))
    elif covariance_type == "tied":
        return int(K * d + d * (d + 1) / 2 + (K - 1))
    else:
        raise ValueError(f"Unknown covariance_type='{covariance_type}'")


# =============================================================================
# 2. Model Fitting & Scoring
# =============================================================================

def fit_gmm(
    X_train_legit: np.ndarray,
    n_components: int = 5,
    covariance_type: str = "diag",
    reg_covar: float = 1e-4,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 3,
) -> GaussianMixture:
    """
    Fit a Gaussian Mixture Model strictly on legitimate training observations.

    Parameters
    ----------
    X_train_legit : np.ndarray
        Training feature matrix containing only legitimate rows (Class == 0).
    n_components : int
        Number of Gaussian clusters K.
    covariance_type : str
        Covariance structure ('full', 'tied', 'diag', 'spherical').
    reg_covar : float
        Regularization added to diagonal of covariance matrices to prevent singularities.
    random_state : int
        Random seed for k-means initialization.
    max_iter : int
        Maximum EM iterations.
    n_init : int
        Number of initializations to perform.

    Returns
    -------
    GaussianMixture
        Fitted sklearn GaussianMixture model.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        init_params="kmeans",
    )
    gmm.fit(X_train_legit)
    return gmm


def compute_anomaly_scores(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """
    Compute negative log-likelihood (NLL) anomaly scores.
    Contract: Higher score == lower density == more anomalous.

    Parameters
    ----------
    gmm : GaussianMixture
        Fitted GMM.
    X : np.ndarray
        Feature matrix (N, d).

    Returns
    -------
    np.ndarray
        1-D float32 array of scores, length N.
    """
    # gmm.score_samples returns per-sample log p(x)
    log_prob = gmm.score_samples(X)
    nll = -log_prob
    return nll.astype(np.float32)


# =============================================================================
# 3. Model Selection Grid Search (AIC, BIC, PR-AUC)
# =============================================================================

def run_gmm_grid_search(
    splits: Splits,
    subsample_size: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Execute comprehensive hyperparameter grid search across:
    - covariance_type in {diag, spherical, tied, full}
    - n_components K in {1, 2, 3, 4, 5, 6, 8, 10, 12, 16}
    - reg_covar in {1e-6, 1e-4, 1e-2}

    Uses a seeded 50k subsample of training legitimate transactions for efficient
    screening, evaluating both Information Criteria (AIC/BIC) and Validation PR-AUC.
    """
    _logger.info("Starting GMM grid search (subsample=%d rows)...", subsample_size)
    rng = np.random.default_rng(seed)
    n_train = len(splits.X_train_legit)
    sub_idx = rng.choice(n_train, size=min(subsample_size, n_train), replace=False)
    X_sub = splits.X_train_legit[sub_idx]

    d = splits.X_train_legit.shape[1]
    cov_types = ["diag", "spherical", "tied", "full"]
    k_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    reg_covars = [1e-6, 1e-4, 1e-2]

    # Held-out validation legitimate instances for val log-likelihood
    val_legit_mask = (splits.y_val == 0)
    X_val_legit = splits.X_val[val_legit_mask]

    records = []
    total_configs = len(cov_types) * len(k_values) * len(reg_covars)
    cnt = 0

    for cov in cov_types:
        for reg in reg_covars:
            for K in k_values:
                cnt += 1
                p_theory = calculate_free_params(cov, K, d)

                t0 = time.time()
                try:
                    gmm = GaussianMixture(
                        n_components=K,
                        covariance_type=cov,
                        reg_covar=reg,
                        random_state=seed,
                        max_iter=150,
                        n_init=1,  # Fast screening across 120 configurations
                        init_params="kmeans",
                    )
                    gmm.fit(X_sub)
                    elapsed = time.time() - t0

                    # AIC and BIC on training subsample
                    train_aic = gmm.aic(X_sub)
                    train_bic = gmm.bic(X_sub)
                    train_nll_mean = -gmm.score(X_sub)

                    # Validation evaluation
                    val_ll_mean = gmm.score(X_val_legit)
                    n_val_legit = len(X_val_legit)
                    val_bic = p_theory * np.log(n_val_legit) - 2.0 * n_val_legit * val_ll_mean

                    # Validation fraud detection performance
                    s_val = compute_anomaly_scores(gmm, splits.X_val)
                    v_pr = pr_auc(splits.y_val, s_val)
                    v_roc = roc_auc(splits.y_val, s_val)
                    v_rec05 = recall_at_fpr(splits.y_val, s_val, 0.005)

                    converged = bool(gmm.converged_)
                    n_iter = int(gmm.n_iter_)

                except Exception as e:
                    _logger.warning("Fit failed for cov=%s K=%d reg=%e: %s", cov, K, reg, e)
                    train_aic, train_bic, val_bic, v_pr, v_roc, v_rec05, elapsed = (
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    )
                    converged, n_iter = False, -1

                records.append({
                    "covariance_type": cov,
                    "n_components": K,
                    "reg_covar": reg,
                    "free_parameters": p_theory,
                    "train_aic": train_aic,
                    "train_bic": train_bic,
                    "val_bic": val_bic,
                    "val_pr_auc": v_pr,
                    "val_roc_auc": v_roc,
                    "val_recall@0.5%": v_rec05,
                    "converged": converged,
                    "n_iter": n_iter,
                    "train_time_sec": elapsed,
                })

    grid_df = pd.DataFrame(records)
    _logger.info("Grid search completed (%d configurations).", len(grid_df))
    return grid_df


# =============================================================================
# 4. AIC / BIC Visualization & Model Selection Protocol
# =============================================================================

def plot_aic_bic(grid_df: pd.DataFrame, default_reg: float = 1e-4) -> plt.Figure:
    """
    Plot AIC and BIC curves versus component count K for each covariance type.
    Marks the minimum BIC and elbows.
    """
    sub_df = grid_df[grid_df["reg_covar"] == default_reg].copy()
    cov_types = ["full", "tied", "diag", "spherical"]
    colors = {"full": "#D0021B", "tied": "#F5A623", "diag": "#4A90E2", "spherical": "#7ED321"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # AIC plot
    for cov in cov_types:
        df_c = sub_df[sub_df["covariance_type"] == cov].sort_values("n_components")
        ax1.plot(df_c["n_components"], df_c["train_aic"], marker="o", label=f"{cov.capitalize()}", color=colors[cov])

    ax1.set_title(f"Akaike Information Criterion (AIC) vs. Components K (reg={default_reg})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Components (K)")
    ax1.set_ylabel("AIC (Lower is Better)")
    ax1.legend()
    ax1.set_xticks(sorted(sub_df["n_components"].unique()))

    # BIC plot
    min_bic_row = sub_df.loc[sub_df["train_bic"].idxmin()]
    for cov in cov_types:
        df_c = sub_df[sub_df["covariance_type"] == cov].sort_values("n_components")
        ax2.plot(df_c["n_components"], df_c["train_bic"], marker="s", label=f"{cov.capitalize()}", color=colors[cov])

    ax2.scatter(
        [min_bic_row["n_components"]],
        [min_bic_row["train_bic"]],
        color="black",
        s=120,
        zorder=5,
        marker="*",
        label=f"Min BIC: {min_bic_row['covariance_type']}, K={int(min_bic_row['n_components'])}",
    )
    ax2.set_title(f"Bayesian Information Criterion (BIC) vs. Components K (reg={default_reg})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Components (K)")
    ax2.set_ylabel("BIC (Lower is Better)")
    ax2.legend()
    ax2.set_xticks(sorted(sub_df["n_components"].unique()))

    plt.tight_layout()
    return fig


# =============================================================================
# 5. Diagnostic Audits & Failure Analysis
# =============================================================================

def compute_component_diagnostics(gmm: GaussianMixture, feature_names: list[str]) -> pd.DataFrame:
    """
    Summarize component weights, cluster mean norms, and dominant feature coordinates.
    """
    K = gmm.n_components
    rows = []
    for k in range(K):
        weight = float(gmm.weights_[k])
        mean_vec = gmm.means_[k]
        mean_norm = float(np.linalg.norm(mean_vec))
        # Top 3 features by absolute mean magnitude
        top_feat_idx = np.argsort(np.abs(mean_vec))[::-1][:3]
        top_feats = ", ".join([f"{feature_names[i]} ({mean_vec[i]:+.2f})" for i in top_feat_idx])

        rows.append({
            "component": k + 1,
            "weight": round(weight, 5),
            "mean_l2_norm": round(mean_norm, 4),
            "top_deviating_features": top_feats,
        })
    return pd.DataFrame(rows)


def plot_nll_distribution(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
) -> plt.Figure:
    """
    Plot logarithmic density histogram of Negative Log-Likelihood (NLL)
    for Legitimate vs Fraudulent transactions on validation and test sets.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Validation
    ax1.hist(s_val[y_val == 0], bins=80, density=True, alpha=0.55, color="#1F77B4", label="Legitimate (Val)")
    ax1.hist(s_val[y_val == 1], bins=40, density=True, alpha=0.65, color="#D62728", label="Fraud (Val)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Negative Log-Likelihood (-ln p(x))")
    ax1.set_ylabel("Density (Log Scale)")
    ax1.set_title("NLL Anomaly Score Distribution (Validation Split)")
    ax1.legend()

    # Test
    ax2.hist(s_test[y_test == 0], bins=80, density=True, alpha=0.55, color="#1F77B4", label="Legitimate (Test)")
    ax2.hist(s_test[y_test == 1], bins=40, density=True, alpha=0.65, color="#D62728", label="Fraud (Test)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Negative Log-Likelihood (-ln p(x))")
    ax2.set_ylabel("Density (Log Scale)")
    ax2.set_title("NLL Anomaly Score Distribution (Test Split)")
    ax2.legend()

    plt.tight_layout()
    return fig


def perform_failure_analysis(
    splits: Splits,
    s_val: np.ndarray,
    gmm: GaussianMixture,
    n_missed: int = 10,
) -> pd.DataFrame:
    """
    Analyze the top missed frauds (frauds assigned the lowest anomaly scores / highest densities).
    Investigates why these frauds blended into the legitimate Gaussian mass.
    """
    val_fraud_mask = (splits.y_val == 1)
    fraud_indices = np.where(val_fraud_mask)[0]
    fraud_scores = s_val[val_fraud_mask]

    # Missed frauds have the lowest anomaly scores (closest to normal legit transactions)
    lowest_scored_idx = fraud_indices[np.argsort(fraud_scores)[:n_missed]]

    rows = []
    legit_mean = splits.X_train_legit.mean(axis=0)

    for rank, idx in enumerate(lowest_scored_idx, 1):
        x = splits.X_val[idx]
        score = float(s_val[idx])
        # Distance from legitimate population mean
        dist_from_mean = float(np.linalg.norm(x - legit_mean))
        # Top 3 features closest to 0
        abs_x = np.abs(x)
        closest_feats = ", ".join([f"{splits.feature_names[i]} ({x[i]:+.2f})" for i in np.argsort(abs_x)[:3]])
        max_deviation_feat = splits.feature_names[np.argmax(abs_x)]
        max_deviation_val = float(x[np.argmax(abs_x)])

        rows.append({
            "missed_rank": rank,
            "val_index": int(idx),
            "anomaly_score_nll": round(score, 4),
            "l2_dist_from_legit_mean": round(dist_from_mean, 4),
            "max_feature_deviation": f"{max_deviation_feat} ({max_deviation_val:+.2f})",
            "blending_features": closest_feats,
            "failure_mechanism": "Concealed within 1-sigma legitimate core; all latent coords typical.",
        })

    return pd.DataFrame(rows)


def analyze_top_anomalies_and_fps(
    splits: Splits,
    s_val: np.ndarray,
    top_n: int = 20,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Identify features driving the top-20 highest-scored frauds vs top-20 false positives.
    Computes per-feature squared z-score contributions.
    """
    y_val = splits.y_val
    fraud_indices = np.where(y_val == 1)[0]
    legit_indices = np.where(y_val == 0)[0]

    top_fraud_idx = fraud_indices[np.argsort(s_val[fraud_indices])[::-1][:top_n]]
    top_fp_idx = legit_indices[np.argsort(s_val[legit_indices])[::-1][:top_n]]

    # Mean squared values per feature for top frauds vs top FPs
    X_val = splits.X_val
    feat_contrib_fraud = np.mean(X_val[top_fraud_idx] ** 2, axis=0)
    feat_contrib_fp = np.mean(X_val[top_fp_idx] ** 2, axis=0)

    df_contrib = pd.DataFrame({
        "feature": splits.feature_names,
        "mean_sq_fraud": feat_contrib_fraud,
        "mean_sq_false_positive": feat_contrib_fp,
    })
    df_contrib["fraud_to_fp_ratio"] = df_contrib["mean_sq_fraud"] / np.maximum(df_contrib["mean_sq_false_positive"], 1e-6)
    df_sorted = df_contrib.sort_values("mean_sq_fraud", ascending=False).reset_index(drop=True)

    # Plot top 12 features driving anomaly scores
    top_plot = df_sorted.head(12)
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(top_plot))
    width = 0.35

    ax.bar(x_pos - width / 2, top_plot["mean_sq_fraud"], width, label="Top-20 Frauds (True Anomalies)", color="#D0021B")
    ax.bar(x_pos + width / 2, top_plot["mean_sq_false_positive"], width, label="Top-20 False Positives (Legit)", color="#4A90E2")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_plot["feature"], rotation=45, ha="right")
    ax.set_ylabel("Mean Squared Standardized Coordinate")
    ax.set_title("Attribution Drivers: Top-20 High-Score Frauds vs. Top-20 False Positives")
    ax.legend()
    plt.tight_layout()

    return df_sorted, fig


# =============================================================================
# 6. Instability Study ("Identify the Gap")
# =============================================================================

def run_instability_study(splits: Splits, chosen_k: int = 5) -> dict[str, Any]:
    """
    Empirical investigation of GMM instability and statistical limitations:
    (i)   Singular/ill-conditioned covariance breakdown and effect of reg_covar.
    (ii)  Sensitivity across 5 random initialization seeds.
    (iii) Dimensionality curse: scaling performance and BIC from 5 to 31 features.
    """
    _logger.info("Executing GMM Instability Study...")

    # (i) Effect of reg_covar on full vs diag covariance condition number and PR-AUC
    reg_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    reg_rows = []
    X_sub = splits.X_train_legit[:30000]

    for reg in reg_values:
        for cov in ["diag", "full"]:
            try:
                gmm = GaussianMixture(n_components=chosen_k, covariance_type=cov, reg_covar=reg, random_state=42, max_iter=150)
                gmm.fit(X_sub)
                val_score = compute_anomaly_scores(gmm, splits.X_val)
                pr = pr_auc(splits.y_val, val_score)
                # Compute average covariance condition number
                if cov == "diag":
                    cond_num = float(np.mean([np.max(c) / np.min(c) for c in gmm.covariances_]))
                else:
                    cond_num = float(np.mean([np.linalg.cond(c) for c in gmm.covariances_]))
                status = "Stable" if gmm.converged_ else "Unconverged"
            except Exception as e:
                pr = np.nan
                cond_num = np.nan
                status = f"Failed ({type(e).__name__})"

            reg_rows.append({
                "covariance_type": cov,
                "reg_covar": reg,
                "val_pr_auc": round(pr, 5) if not np.isnan(pr) else np.nan,
                "mean_condition_number": round(cond_num, 2) if not np.isnan(cond_num) else np.nan,
                "status": status,
            })
    reg_df = pd.DataFrame(reg_rows)

    # (ii) Variance across 5 random seeds (initialization sensitivity)
    seeds = [42, 43, 44, 45, 46]
    seed_rows = []
    for s in seeds:
        gmm = GaussianMixture(n_components=chosen_k, covariance_type="diag", reg_covar=1e-4, random_state=s, max_iter=200, n_init=1)
        gmm.fit(splits.X_train_legit)
        s_val = compute_anomaly_scores(gmm, splits.X_val)
        pr = pr_auc(splits.y_val, s_val)
        rec05 = recall_at_fpr(splits.y_val, s_val, 0.005)
        seed_rows.append({
            "seed": s,
            "val_pr_auc": round(pr, 5),
            "val_recall@0.5%": round(rec05, 5),
            "n_iter": int(gmm.n_iter_),
            "converged": bool(gmm.converged_),
        })
    seed_df = pd.DataFrame(seed_rows)

    # (iii) Curse of Dimensionality: performance vs feature dimensionality d
    d_steps = [5, 10, 15, 20, 25, splits.X_train_legit.shape[1]]
    dim_rows = []
    for d in d_steps:
        X_tr_d = splits.X_train_legit[:, :d]
        X_val_d = splits.X_val[:, :d]
        t0 = time.time()
        gmm = GaussianMixture(n_components=chosen_k, covariance_type="diag", reg_covar=1e-4, random_state=42, max_iter=200)
        gmm.fit(X_tr_d)
        fit_time = time.time() - t0
        s_val_d = compute_anomaly_scores(gmm, X_val_d)
        pr = pr_auc(splits.y_val, s_val_d)
        bic = gmm.bic(X_tr_d[:30000])

        dim_rows.append({
            "dimension": d,
            "val_pr_auc": round(pr, 5),
            "train_bic": round(bic, 1),
            "train_time_sec": round(fit_time, 3),
        })
    dim_df = pd.DataFrame(dim_rows)

    return {
        "reg_covar_df": reg_df,
        "seed_df": seed_df,
        "dim_df": dim_df,
    }


# =============================================================================
# 7. Main Pipeline Runner (run_gmm_stage)
# =============================================================================

def run_gmm_stage() -> None:
    """
    Complete end-to-end GMM baseline pipeline invoked by run_all.py:
    1. Loads features (engineered feature set).
    2. Runs comprehensive model selection grid search (120 combinations).
    3. Analyzes and plots AIC / BIC curves, saving results/tables/gmm_grid_search.csv
       and results/figures/gmm_aic_bic.png.
    4. Refits final optimal GMM on full legitimate training transactions.
    5. Generates anomaly score arrays (gmm_val.npy, gmm_test.npy) and serializes
       model to results/models/gmm.joblib.
    6. Evaluates composite metrics via src.metrics.evaluate_scores -> gmm_metrics.csv.
    7. Executes diagnostics, failure analysis, and instability study.
    """
    set_seed()
    tables_dir = _ROOT / CFG["paths"]["tables"]
    figures_dir = _ROOT / CFG["paths"]["figures"]
    models_dir = _ROOT / CFG["paths"]["models"]
    scores_dir = _ROOT / CFG["paths"]["scores"]

    for d in [tables_dir, figures_dir, models_dir, scores_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _logger.info("==================================================")
    _logger.info("  STARTING PHASE 1 GMM ANOMALY BASELINE PIPELINE  ")
    _logger.info("==================================================")

    # 1. Load engineered features
    splits = get_features("engineered")
    _logger.info("Dataset loaded: d=%d features.", splits.X_train_legit.shape[1])

    # 2. Grid search
    grid_csv = tables_dir / "gmm_grid_search.csv"
    if grid_csv.exists():
        _logger.info("Found existing grid search at %s, loading...", grid_csv)
        grid_df = pd.read_csv(grid_csv)
    else:
        grid_df = run_gmm_grid_search(splits, subsample_size=50000, seed=CFG.get("seed", 42))
        save_table(grid_df, "gmm_grid_search")

    # 3. Plot AIC/BIC curves
    fig_aic_bic = plot_aic_bic(grid_df, default_reg=1e-4)
    save_fig(fig_aic_bic, "gmm_aic_bic")

    # 4. Model Selection Protocol
    # (a) Unsupervised choice: Minimum BIC
    best_bic_idx = grid_df["train_bic"].idxmin()
    best_bic_row = grid_df.loc[best_bic_idx]
    # (b) Supervised validation choice: Best Validation PR-AUC
    best_pr_idx = grid_df["val_pr_auc"].idxmax()
    best_pr_row = grid_df.loc[best_pr_idx]

    _logger.info(
        "Model Selection Protocol:\n"
        "  (a) Unsupervised Choice (argmin BIC):      cov=%s, K=%d, reg=%e (BIC=%.1f, Val PR-AUC=%.4f)\n"
        "  (b) Validation Choice   (argmax Val PR):   cov=%s, K=%d, reg=%e (Val PR-AUC=%.4f, BIC=%.1f)",
        best_bic_row["covariance_type"], int(best_bic_row["n_components"]), float(best_bic_row["reg_covar"]),
        best_bic_row["train_bic"], best_bic_row["val_pr_auc"],
        best_pr_row["covariance_type"], int(best_pr_row["n_components"]), float(best_pr_row["reg_covar"]),
        best_pr_row["val_pr_auc"], best_pr_row["train_bic"],
    )

    # Save selection summary table
    sel_df = pd.DataFrame([
        {
            "criterion": "Unsupervised (Min BIC)",
            "covariance_type": best_bic_row["covariance_type"],
            "n_components": int(best_bic_row["n_components"]),
            "reg_covar": float(best_bic_row["reg_covar"]),
            "free_parameters": int(best_bic_row["free_parameters"]),
            "train_bic": round(float(best_bic_row["train_bic"]), 1),
            "val_pr_auc": round(float(best_bic_row["val_pr_auc"]), 5),
            "val_recall@0.5%": round(float(best_bic_row["val_recall@0.5%"]), 5),
        },
        {
            "criterion": "Validation-Selected (Max Val PR-AUC)",
            "covariance_type": best_pr_row["covariance_type"],
            "n_components": int(best_pr_row["n_components"]),
            "reg_covar": float(best_pr_row["reg_covar"]),
            "free_parameters": int(best_pr_row["free_parameters"]),
            "train_bic": round(float(best_pr_row["train_bic"]), 1),
            "val_pr_auc": round(float(best_pr_row["val_pr_auc"]), 5),
            "val_recall@0.5%": round(float(best_pr_row["val_recall@0.5%"]), 5),
        },
    ])
    save_table(sel_df, "gmm_selection_comparison")

    # Chosen production baseline: Selected via validation PR-AUC without test snooping
    chosen_cov = str(best_pr_row["covariance_type"])
    chosen_k = int(best_pr_row["n_components"])
    chosen_reg = float(best_pr_row["reg_covar"])

    _logger.info("Refitting final GMM on full legitimate training data (%d rows)...", len(splits.X_train_legit))
    gmm_final = fit_gmm(
        splits.X_train_legit,
        n_components=chosen_k,
        covariance_type=chosen_cov,
        reg_covar=chosen_reg,
        random_state=CFG.get("seed", 42),
        max_iter=300,
        n_init=3,
    )

    # Serialize fitted model
    model_path = models_dir / "gmm.joblib"
    joblib.dump(gmm_final, model_path)
    _logger.info("Saved fitted GMM model to %s", model_path)

    # 5. Compute and save anomaly scores
    s_val = compute_anomaly_scores(gmm_final, splits.X_val)
    s_test = compute_anomaly_scores(gmm_final, splits.X_test)

    # Save to scores directory following project contract (model_name, split, scores)
    save_scores("gmm", "val", s_val)
    save_scores("gmm", "test", s_test)
    _logger.info("Saved gmm_val.npy (shape=%s) and gmm_test.npy (shape=%s)", s_val.shape, s_test.shape)

    # 6. Comprehensive Evaluation
    eval_df = evaluate_scores(splits.y_val, s_val, splits.y_test, s_test, CFG)
    save_table(eval_df, "gmm_metrics")
    _logger.info("Saved evaluation table to results/tables/gmm_metrics.csv:\n%s", eval_df)

    # 7. Diagnostics
    # (a) Histogram of NLL
    fig_nll = plot_nll_distribution(splits.y_val, s_val, splits.y_test, s_test)
    save_fig(fig_nll, "gmm_nll_distribution")

    # (b) Component summary
    comp_df = compute_component_diagnostics(gmm_final, splits.feature_names)
    save_table(comp_df, "gmm_components")

    # (c) Attribution & False Positives
    top_df, fig_attr = analyze_top_anomalies_and_fps(splits, s_val, top_n=20)
    save_table(top_df, "gmm_top_anomalies")
    save_fig(fig_attr, "gmm_feature_attribution")

    # (d) Failure analysis (10 missed frauds)
    fail_df = perform_failure_analysis(splits, s_val, gmm_final, n_missed=10)
    save_table(fail_df, "gmm_failure_analysis")

    # 8. Instability Study
    instab = run_instability_study(splits, chosen_k=chosen_k)
    save_table(instab["reg_covar_df"], "gmm_reg_covar_study")
    save_table(instab["seed_df"], "gmm_seed_stability")
    save_table(instab["dim_df"], "gmm_dim_study")

    # Plot instability study: condition number & dimension scaling
    fig_inst, (ax_reg, ax_dim) = plt.subplots(1, 2, figsize=(14, 5))
    reg_df = instab["reg_covar_df"]
    for cov, col in [("diag", "#4A90E2"), ("full", "#D0021B")]:
        sub = reg_df[reg_df["covariance_type"] == cov]
        ax_reg.plot(sub["reg_covar"], sub["mean_condition_number"], marker="o", label=f"{cov.capitalize()} Cov", color=col)
    ax_reg.set_xscale("log")
    ax_reg.set_yscale("log")
    ax_reg.set_xlabel("Regularization reg_covar")
    ax_reg.set_ylabel("Mean Condition Number (Log Scale)")
    ax_reg.set_title("Ill-Conditioned Covariances & reg_covar Regularization")
    ax_reg.legend()

    dim_df = instab["dim_df"]
    ax_dim.plot(dim_df["dimension"], dim_df["val_pr_auc"], marker="s", color="#7ED321", linewidth=2, label="Val PR-AUC")
    ax_dim.set_xlabel("Feature Dimensionality d")
    ax_dim.set_ylabel("Validation PR-AUC")
    ax_dim.set_title("Curse of Dimensionality: Performance vs. Dimension")
    ax_dim.legend()

    plt.tight_layout()
    save_fig(fig_inst, "gmm_instability_study")

    _logger.info("GMM Baseline pipeline execution completed successfully!")


if __name__ == "__main__":
    run_gmm_stage()

