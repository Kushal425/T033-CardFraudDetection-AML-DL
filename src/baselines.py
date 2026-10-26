"""
src/baselines.py
================
Classical anomaly-detection baselines for Team T033 — Card Fraud Triage.
Author: Vani Rudra (Task 03 — Phase 1: ML only).

Models implemented (all fit on **training-legitimate rows only**):
  1. PCA reconstruction error  — score = ‖x − x̂‖²
  2. Isolation Forest           — score = −score_samples(X)
  3. One-Class SVM (RBF)        — score = −decision_function(X)
  4. Local Outlier Factor        — score = −score_samples(X)  (novelty=True)
  5. Random scores               — sanity-check reference
  6. |Amount|-only score         — simple reference baseline

Score convention: **higher == more anomalous** (consistent with src/metrics.py).

Leakage rules:
- Models are fit on X_train_legit (Class==0 training rows only).
- Fraud labels are used ONLY for hyperparameter/threshold selection on VAL
  and for evaluation via src/metrics.py.
- Test is touched exactly once for final evaluation.
- OC-SVM and LOF use seeded sub-samples of train-legit (O(n²) complexity).

Seed: all randomness uses CFG['seed'] = 42 for reproducibility.
"""

from __future__ import annotations

import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any

import joblib
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.config import CFG, get_repo_root
from src.data import get_features, Splits
from src.metrics import evaluate_scores, pr_auc
from src.utils import (
    get_logger,
    save_fig,
    save_scores,
    save_table,
    set_seed,
    load_scores,
)

_logger = get_logger(__name__)
_ROOT: Path = get_repo_root()
_SEED: int = int(CFG.get("seed", 42))
_MODELS_DIR: Path = _ROOT / CFG["paths"].get("models", "results/models")


# ═══════════════════════════════════════════════════════════════════════════
# 0. Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in [
        _ROOT / CFG["paths"]["scores"],
        _ROOT / CFG["paths"]["figures"],
        _ROOT / CFG["paths"]["tables"],
        _MODELS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def _timed(func):
    """Decorator to time a function and track peak memory."""
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak_mb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb /= 1024 ** 2
        _logger.info(
            "%s finished in %.1f s (peak mem %.1f MB)",
            func.__name__, elapsed, peak_mb,
        )
        return result, elapsed, peak_mb
    return wrapper


def _subsample_train_legit(
    X: np.ndarray, n: int, seed: int = _SEED
) -> np.ndarray:
    """
    Return a seeded random sub-sample of X (without replacement).

    Used for OC-SVM and LOF, which have O(n²) or O(n·k) fitting cost.
    We document the size and reason.
    """
    rng = np.random.RandomState(seed)
    if n >= len(X):
        _logger.info("Subsample size %d >= dataset size %d; using full dataset.", n, len(X))
        return X.copy()
    idx = rng.choice(len(X), size=n, replace=False)
    _logger.info("Sub-sampled %d / %d rows (seed=%d) for fitting.", n, len(X), seed)
    return X[idx]


# ═══════════════════════════════════════════════════════════════════════════
# 1. PCA Reconstruction Error
# ═══════════════════════════════════════════════════════════════════════════

def fit_pca(X_train_legit: np.ndarray, n_components: int) -> PCA:
    """Fit PCA on training-legitimate rows."""
    pca = PCA(n_components=n_components, random_state=_SEED)
    pca.fit(X_train_legit)
    return pca


def score_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    """
    Compute PCA reconstruction error: score = ‖x − x̂‖².

    Higher reconstruction error → more anomalous.
    """
    X_hat = pca.inverse_transform(pca.transform(X))
    recon_error = np.sum((X - X_hat) ** 2, axis=1)
    return recon_error.astype(np.float32)


def sweep_pca(
    X_train_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Sweep over PCA component counts.

    Strategy:
    - Variance targets: find k for 90%, 95%, 99% explained variance.
    - Full sweep: k ∈ {1, 2, 3, ..., d} (capped at 31 features).
    - Selection: choose k with best validation PR-AUC.
    """
    d = X_train_legit.shape[1]

    # Full PCA for explained variance curve
    pca_full = PCA(random_state=_SEED).fit(X_train_legit)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    # Variance target k values
    var_targets = {0.90: None, 0.95: None, 0.99: None}
    for target in var_targets:
        idx = np.searchsorted(cumvar, target)
        var_targets[target] = int(idx + 1)  # 1-indexed
    _logger.info("Variance targets: %s", var_targets)

    # Full k-sweep
    k_values = sorted(set(list(range(1, d + 1)) + list(var_targets.values())))
    results = []

    for k in k_values:
        pca_k = fit_pca(X_train_legit, n_components=k)
        s_val = score_pca(pca_k, X_val)
        prauc = pr_auc(y_val, s_val)
        results.append({
            "k": k,
            "explained_variance": float(cumvar[k - 1]),
            "val_pr_auc": prauc,
        })
        _logger.info("PCA k=%d  var=%.4f  val_PR-AUC=%.4f", k, cumvar[k - 1], prauc)

    results_df = pd.DataFrame(results)

    # Best k
    best_row = results_df.loc[results_df["val_pr_auc"].idxmax()]
    best_k = int(best_row["k"])
    _logger.info(
        "Best PCA k=%d  (val_PR-AUC=%.4f, explained_variance=%.4f)",
        best_k, best_row["val_pr_auc"], best_row["explained_variance"],
    )

    return {
        "results_df": results_df,
        "best_k": best_k,
        "cumvar": cumvar,
        "var_targets": var_targets,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Isolation Forest
# ═══════════════════════════════════════════════════════════════════════════

def fit_iforest(
    X_train_legit: np.ndarray,
    n_estimators: int = 100,
    max_samples: int | str = "auto",
    max_features: float = 1.0,
) -> IsolationForest:
    """
    Fit Isolation Forest on training-legitimate rows.

    contamination is NOT used for fitting (set to 'auto' which defaults to 0.5
    threshold — we ignore decision_function thresholding and use score_samples).
    """
    ifo = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination="auto",
        random_state=_SEED,
        n_jobs=-1,
    )
    ifo.fit(X_train_legit)
    return ifo


def score_iforest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Anomaly score = −score_samples(X).

    sklearn's score_samples returns the negative anomaly score of the input;
    we negate it so that higher == more anomalous.
    """
    return (-model.score_samples(X)).astype(np.float32)


def sweep_iforest(
    X_train_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Grid search over Isolation Forest hyperparameters.

    Grid:
    - n_estimators ∈ {100, 200, 500}
    - max_samples ∈ {256, 1024, 'auto'}
    - max_features ∈ {0.5, 1.0}

    Selection: best validation PR-AUC.
    """
    grid = []
    for n_est in [100, 200, 500]:
        for ms in [256, 1024, "auto"]:
            for mf in [0.5, 1.0]:
                grid.append({
                    "n_estimators": n_est,
                    "max_samples": ms,
                    "max_features": mf,
                })

    results = []
    for params in grid:
        t0 = time.perf_counter()
        model = fit_iforest(X_train_legit, **params)
        fit_time = time.perf_counter() - t0
        s_val = score_iforest(model, X_val)
        prauc = pr_auc(y_val, s_val)
        results.append({**params, "val_pr_auc": prauc, "fit_time_s": fit_time})
        _logger.info(
            "IF n_est=%d ms=%s mf=%.1f → val_PR-AUC=%.4f (%.1fs)",
            params["n_estimators"], params["max_samples"],
            params["max_features"], prauc, fit_time,
        )

    results_df = pd.DataFrame(results)
    best_idx = results_df["val_pr_auc"].idxmax()
    best_params = results_df.loc[best_idx]
    _logger.info("Best IF: %s", best_params.to_dict())

    return {"results_df": results_df, "best_params": best_params.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════
# 3. One-Class SVM (RBF kernel)
# ═══════════════════════════════════════════════════════════════════════════

# OC-SVM has O(n² · d) to O(n³ · d) fitting complexity, so we sub-sample
# training-legitimate rows to 40,000.  This is documented in the viva notes.
_OCSVM_SUBSAMPLE_SIZE = 40_000


def fit_ocsvm(
    X_train_legit: np.ndarray,
    nu: float = 0.01,
    gamma: str | float = "scale",
    subsample_n: int = _OCSVM_SUBSAMPLE_SIZE,
) -> OneClassSVM:
    """
    Fit One-Class SVM (RBF kernel) on a seeded sub-sample of train-legit.

    ν bounds the fraction of outliers and support vectors.
    """
    X_sub = _subsample_train_legit(X_train_legit, subsample_n)
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    ocsvm.fit(X_sub)
    return ocsvm


def score_ocsvm(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """
    Anomaly score = −decision_function(X).

    decision_function returns positive values for inliers; we negate so
    higher == more anomalous.
    """
    return (-model.decision_function(X)).astype(np.float32)


def sweep_ocsvm(
    X_train_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Grid search over OC-SVM hyperparameters.

    Grid:
    - ν ∈ {0.001, 0.01, 0.05}
    - γ ∈ {'scale', 0.01, 0.1}

    Selection: best validation PR-AUC.
    """
    grid = []
    for nu in [0.001, 0.01, 0.05]:
        for gamma in ["scale", 0.01, 0.1]:
            grid.append({"nu": nu, "gamma": gamma})

    results = []
    for params in grid:
        t0 = time.perf_counter()
        model = fit_ocsvm(X_train_legit, **params)
        fit_time = time.perf_counter() - t0
        s_val = score_ocsvm(model, X_val)
        prauc = pr_auc(y_val, s_val)
        results.append({**params, "val_pr_auc": prauc, "fit_time_s": fit_time})
        _logger.info(
            "OCSVM nu=%.3f gamma=%s → val_PR-AUC=%.4f (%.1fs)",
            params["nu"], params["gamma"], prauc, fit_time,
        )

    results_df = pd.DataFrame(results)
    best_idx = results_df["val_pr_auc"].idxmax()
    best_params = results_df.loc[best_idx]
    _logger.info("Best OCSVM: %s", best_params.to_dict())

    return {"results_df": results_df, "best_params": best_params.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Local Outlier Factor (novelty mode)
# ═══════════════════════════════════════════════════════════════════════════

# LOF has O(n · k · d) fitting cost with k-nearest-neighbor queries.
# We use a sub-sample for fitting to keep runtime manageable.
_LOF_SUBSAMPLE_SIZE = 40_000


def fit_lof(
    X_train_legit: np.ndarray,
    n_neighbors: int = 20,
    subsample_n: int = _LOF_SUBSAMPLE_SIZE,
) -> LocalOutlierFactor:
    """
    Fit LOF in novelty mode on a seeded sub-sample of train-legit.

    novelty=True allows calling score_samples on new data.
    """
    X_sub = _subsample_train_legit(X_train_legit, subsample_n)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(X_sub)
    return lof


def score_lof(model: LocalOutlierFactor, X: np.ndarray) -> np.ndarray:
    """
    Anomaly score = −score_samples(X).

    LOF's score_samples returns negative anomaly scores; negate for
    higher == more anomalous.
    """
    return (-model.score_samples(X)).astype(np.float32)


def sweep_lof(
    X_train_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Sweep over LOF hyperparameters.

    Grid: n_neighbors ∈ {10, 20, 50}
    Selection: best validation PR-AUC.
    """
    results = []
    for nn in [10, 20, 50]:
        t0 = time.perf_counter()
        model = fit_lof(X_train_legit, n_neighbors=nn)
        fit_time = time.perf_counter() - t0
        s_val = score_lof(model, X_val)
        prauc = pr_auc(y_val, s_val)
        results.append({
            "n_neighbors": nn,
            "val_pr_auc": prauc,
            "fit_time_s": fit_time,
        })
        _logger.info(
            "LOF n_neighbors=%d → val_PR-AUC=%.4f (%.1fs)",
            nn, prauc, fit_time,
        )

    results_df = pd.DataFrame(results)
    best_idx = results_df["val_pr_auc"].idxmax()
    best_params = results_df.loc[best_idx]
    _logger.info("Best LOF: %s", best_params.to_dict())

    return {"results_df": results_df, "best_params": best_params.to_dict()}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Reference baselines
# ═══════════════════════════════════════════════════════════════════════════

def score_random(n: int, seed: int = _SEED) -> np.ndarray:
    """
    Generate uniform random scores in [0, 1].

    PR-AUC of random scores should ≈ base_rate (fraud fraction) — this is
    a sanity check.
    """
    rng = np.random.RandomState(seed)
    return rng.rand(n).astype(np.float32)


def score_amount(splits: Splits, split: str = "val") -> np.ndarray:
    """
    Use |Amount| as the anomaly score.

    The log1p(Amount) feature is already in the feature matrix (index = 28,
    column 'log1p_amount').  We use the raw (unscaled) absolute amount from
    the original data instead to make it interpretable.

    But since we already have scaled features, we use the log1p_amount feature
    (which is standardized) as a proxy.  Higher amount → more suspicious
    (simple heuristic).
    """
    if split == "val":
        X = splits.X_val
    elif split == "test":
        X = splits.X_test
    else:
        raise ValueError(f"Unknown split: {split}")

    # log1p_amount is at index 28 in the base feature set
    # (V1-V28 = indices 0-27, log1p_amount = 28, sin_time=29, cos_time=30)
    amount_idx = splits.feature_names.index("log1p_amount")
    return np.abs(X[:, amount_idx]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Diagnostics & Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_explained_variance(cumvar: np.ndarray, var_targets: dict) -> plt.Figure:
    """Plot cumulative explained variance vs number of components."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "b-o", markersize=4, label="Cumulative variance")
    for target, k in var_targets.items():
        ax.axhline(y=target, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=k, color="gray", linestyle="--", alpha=0.5)
        ax.annotate(
            f"{target*100:.0f}% → k={k}",
            xy=(k, target), fontsize=9,
            xytext=(k + 1, target - 0.03),
        )
    ax.set_xlabel("Number of PCA Components (k)")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA: Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pca_prauc_vs_k(results_df: pd.DataFrame) -> plt.Figure:
    """Plot PR-AUC vs number of PCA components."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["k"], results_df["val_pr_auc"], "r-o", markersize=4)
    best = results_df.loc[results_df["val_pr_auc"].idxmax()]
    ax.axvline(x=best["k"], color="green", linestyle="--", label=f"Best k={int(best['k'])}")
    ax.set_xlabel("Number of PCA Components (k)")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("PCA Reconstruction Error: PR-AUC vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_histograms(
    scores_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    title_prefix: str = "Val",
) -> plt.Figure:
    """
    Plot score distributions for legit vs fraud for each model.

    scores_dict: {model_name: scores_array}
    """
    n_models = len(scores_dict)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 8))
    axes = axes.flatten()

    legit_mask = y == 0
    fraud_mask = y == 1

    for i, (name, scores) in enumerate(scores_dict.items()):
        ax = axes[i]
        ax.hist(scores[legit_mask], bins=80, alpha=0.6, label="Legit", density=True, color="steelblue")
        ax.hist(scores[fraud_mask], bins=80, alpha=0.6, label="Fraud", density=True, color="crimson")
        ax.set_title(f"{title_prefix}: {name}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{title_prefix} Split: Score Distributions (Legit vs Fraud)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_rank_correlation(
    scores_dict: dict[str, np.ndarray],
) -> plt.Figure:
    """
    Compute and plot Spearman rank-correlation matrix between all models.
    """
    import seaborn as sns

    names = list(scores_dict.keys())
    n = len(names)
    corr = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            else:
                rho, _ = scipy_stats.spearmanr(scores_dict[names[i]], scores_dict[names[j]])
                corr[i, j] = rho

    corr_df = pd.DataFrame(corr, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_df, annot=True, fmt=".3f", cmap="RdYlBu_r",
        vmin=-1, vmax=1, ax=ax, square=True,
    )
    ax.set_title("Spearman Rank Correlation Between Anomaly Scores (Val)")
    fig.tight_layout()
    return fig, corr_df


def analyse_top_k(
    scores: np.ndarray,
    y: np.ndarray,
    model_name: str,
    k: int = 20,
) -> pd.DataFrame:
    """
    Analyse top-k highest-scored transactions: how many are fraud vs false positives.
    """
    top_idx = np.argsort(scores)[::-1][:k]
    rows = []
    for rank, idx in enumerate(top_idx):
        rows.append({
            "rank": rank + 1,
            "index": idx,
            "score": float(scores[idx]),
            "is_fraud": int(y[idx]),
            "classification": "TP (caught fraud)" if y[idx] == 1 else "FP (false positive)",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_all_baselines() -> dict:
    """
    Run the full classical baseline pipeline:

    1. Load data via get_features("base").
    2. For each model: hyperparameter sweep → best model → val/test scores.
    3. Reference baselines: random scores, |Amount| score.
    4. Evaluate all via evaluate_scores.
    5. Diagnostics: histograms, rank correlation, top-K analysis.
    6. Save all outputs.

    Returns a dict with all results for downstream use.
    """
    set_seed(_SEED)
    _ensure_dirs()

    _logger.info("=" * 70)
    _logger.info("STARTING CLASSICAL BASELINES PIPELINE")
    _logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    splits = get_features("base")
    _logger.info(
        "Data loaded: X_train_legit=%s, X_val=%s, X_test=%s, features=%d",
        splits.X_train_legit.shape, splits.X_val.shape,
        splits.X_test.shape, len(splits.feature_names),
    )

    all_scores_val = {}
    all_scores_test = {}
    all_grid_tables = {}
    all_metrics = {}
    timing_info = []

    # ══════════════════════════════════════════════════════════════════════
    # 1. PCA Reconstruction Error
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  PCA RECONSTRUCTION ERROR\n" + "=" * 50)

    t0 = time.perf_counter()
    pca_results = sweep_pca(
        splits.X_train_legit, splits.X_val, splits.y_val,
    )
    pca_sweep_time = time.perf_counter() - t0

    # Save PCA sweep table
    save_table(pca_results["results_df"], "baseline_pca_sweep")
    all_grid_tables["pca"] = pca_results["results_df"]

    # Plot explained variance
    fig_var = plot_explained_variance(pca_results["cumvar"], pca_results["var_targets"])
    save_fig(fig_var, "baseline_pca_explained_variance")
    plt.close(fig_var)

    # Plot PR-AUC vs k
    fig_prauc = plot_pca_prauc_vs_k(pca_results["results_df"])
    save_fig(fig_prauc, "baseline_pca_prauc_vs_k")
    plt.close(fig_prauc)

    # Best PCA model
    best_k = pca_results["best_k"]
    pca_model = fit_pca(splits.X_train_legit, n_components=best_k)
    s_pca_val = score_pca(pca_model, splits.X_val)
    s_pca_test = score_pca(pca_model, splits.X_test)
    all_scores_val["pca"] = s_pca_val
    all_scores_test["pca"] = s_pca_test

    # Save model and scores
    joblib.dump(pca_model, _MODELS_DIR / "pca.joblib")
    with open(_MODELS_DIR / "pca.pkl", "wb") as f:
        pickle.dump(pca_model, f)
    save_scores("pca", "val", s_pca_val)
    save_scores("pca", "test", s_pca_test)

    timing_info.append({
        "model": "PCA",
        "sweep_time_s": pca_sweep_time,
        "memory_mb": round((_MODELS_DIR / "pca.joblib").stat().st_size / (1024 * 1024), 3),
        "best_params": f"k={best_k}",
    })

    # ══════════════════════════════════════════════════════════════════════
    # 2. Isolation Forest
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  ISOLATION FOREST\n" + "=" * 50)

    t0 = time.perf_counter()
    iforest_results = sweep_iforest(
        splits.X_train_legit, splits.X_val, splits.y_val,
    )
    iforest_sweep_time = time.perf_counter() - t0

    save_table(iforest_results["results_df"], "baseline_iforest_sweep")
    all_grid_tables["iforest"] = iforest_results["results_df"]

    # Best IF model
    bp = iforest_results["best_params"]
    best_if = fit_iforest(
        splits.X_train_legit,
        n_estimators=int(bp["n_estimators"]),
        max_samples=bp["max_samples"],
        max_features=float(bp["max_features"]),
    )
    s_if_val = score_iforest(best_if, splits.X_val)
    s_if_test = score_iforest(best_if, splits.X_test)
    all_scores_val["iforest"] = s_if_val
    all_scores_test["iforest"] = s_if_test

    joblib.dump(best_if, _MODELS_DIR / "iforest.joblib")
    with open(_MODELS_DIR / "iforest.pkl", "wb") as f:
        pickle.dump(best_if, f)
    save_scores("iforest", "val", s_if_val)
    save_scores("iforest", "test", s_if_test)

    timing_info.append({
        "model": "Isolation Forest",
        "sweep_time_s": iforest_sweep_time,
        "memory_mb": round((_MODELS_DIR / "iforest.joblib").stat().st_size / (1024 * 1024), 3),
        "best_params": f"n_est={int(bp['n_estimators'])}, ms={bp['max_samples']}, mf={bp['max_features']}",
    })

    # ══════════════════════════════════════════════════════════════════════
    # 3. One-Class SVM
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  ONE-CLASS SVM (RBF)\n" + "=" * 50)

    t0 = time.perf_counter()
    ocsvm_results = sweep_ocsvm(
        splits.X_train_legit, splits.X_val, splits.y_val,
    )
    ocsvm_sweep_time = time.perf_counter() - t0

    save_table(ocsvm_results["results_df"], "baseline_ocsvm_sweep")
    all_grid_tables["ocsvm"] = ocsvm_results["results_df"]

    bp = ocsvm_results["best_params"]
    best_ocsvm = fit_ocsvm(
        splits.X_train_legit,
        nu=float(bp["nu"]),
        gamma=bp["gamma"],
    )
    s_ocsvm_val = score_ocsvm(best_ocsvm, splits.X_val)
    s_ocsvm_test = score_ocsvm(best_ocsvm, splits.X_test)
    all_scores_val["ocsvm"] = s_ocsvm_val
    all_scores_test["ocsvm"] = s_ocsvm_test

    joblib.dump(best_ocsvm, _MODELS_DIR / "ocsvm.joblib")
    with open(_MODELS_DIR / "ocsvm.pkl", "wb") as f:
        pickle.dump(best_ocsvm, f)
    save_scores("ocsvm", "val", s_ocsvm_val)
    save_scores("ocsvm", "test", s_ocsvm_test)

    timing_info.append({
        "model": "OC-SVM",
        "sweep_time_s": ocsvm_sweep_time,
        "memory_mb": round((_MODELS_DIR / "ocsvm.joblib").stat().st_size / (1024 * 1024), 3),
        "best_params": f"nu={bp['nu']}, gamma={bp['gamma']}",
    })

    # ══════════════════════════════════════════════════════════════════════
    # 4. Local Outlier Factor
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  LOCAL OUTLIER FACTOR\n" + "=" * 50)

    t0 = time.perf_counter()
    lof_results = sweep_lof(
        splits.X_train_legit, splits.X_val, splits.y_val,
    )
    lof_sweep_time = time.perf_counter() - t0

    save_table(lof_results["results_df"], "baseline_lof_sweep")
    all_grid_tables["lof"] = lof_results["results_df"]

    bp = lof_results["best_params"]
    best_lof = fit_lof(
        splits.X_train_legit,
        n_neighbors=int(bp["n_neighbors"]),
    )
    s_lof_val = score_lof(best_lof, splits.X_val)
    s_lof_test = score_lof(best_lof, splits.X_test)
    all_scores_val["lof"] = s_lof_val
    all_scores_test["lof"] = s_lof_test

    joblib.dump(best_lof, _MODELS_DIR / "lof.joblib")
    with open(_MODELS_DIR / "lof.pkl", "wb") as f:
        pickle.dump(best_lof, f)
    save_scores("lof", "val", s_lof_val)
    save_scores("lof", "test", s_lof_test)

    timing_info.append({
        "model": "LOF",
        "sweep_time_s": lof_sweep_time,
        "memory_mb": round((_MODELS_DIR / "lof.joblib").stat().st_size / (1024 * 1024), 3),
        "best_params": f"n_neighbors={int(bp['n_neighbors'])}",
    })

    # ══════════════════════════════════════════════════════════════════════
    # 5. Reference baselines
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  REFERENCE BASELINES\n" + "=" * 50)

    # Random scores
    s_rand_val = score_random(len(splits.y_val), seed=_SEED)
    s_rand_test = score_random(len(splits.y_test), seed=_SEED + 1)
    all_scores_val["random"] = s_rand_val
    all_scores_test["random"] = s_rand_test
    save_scores("random", "val", s_rand_val)
    save_scores("random", "test", s_rand_test)

    rand_prauc = pr_auc(splits.y_val, s_rand_val)
    base_rate_val = splits.y_val.mean()
    _logger.info(
        "Random baseline: val_PR-AUC=%.6f (should ≈ base_rate=%.6f)",
        rand_prauc, base_rate_val,
    )

    # Amount-only scores
    s_amt_val = score_amount(splits, "val")
    s_amt_test = score_amount(splits, "test")
    all_scores_val["amount"] = s_amt_val
    all_scores_test["amount"] = s_amt_test
    save_scores("amount", "val", s_amt_val)
    save_scores("amount", "test", s_amt_test)

    _logger.info(
        "Amount baseline: val_PR-AUC=%.4f", pr_auc(splits.y_val, s_amt_val)
    )

    # ══════════════════════════════════════════════════════════════════════
    # 6. Evaluate all models
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  EVALUATION\n" + "=" * 50)

    model_names = ["pca", "iforest", "ocsvm", "lof", "random", "amount"]
    all_eval_dfs = []

    for name in model_names:
        eval_df = evaluate_scores(
            splits.y_val, all_scores_val[name],
            splits.y_test, all_scores_test[name],
            CFG,
        )
        eval_df.insert(0, "model", name)
        all_eval_dfs.append(eval_df)
        all_metrics[name] = eval_df

    # Try to include GMM scores if available
    try:
        s_gmm_val = load_scores("gmm", "val")
        s_gmm_test = load_scores("gmm", "test")
        if len(s_gmm_val) == len(splits.y_val) and len(s_gmm_test) == len(splits.y_test):
            eval_gmm = evaluate_scores(
                splits.y_val, s_gmm_val,
                splits.y_test, s_gmm_test,
                CFG,
            )
            eval_gmm.insert(0, "model", "gmm")
            all_eval_dfs.append(eval_gmm)
            all_scores_val["gmm"] = s_gmm_val
            all_scores_test["gmm"] = s_gmm_test
            _logger.info("Included GMM scores in comparison table.")
        else:
            _logger.warning(
                "GMM score lengths mismatch: val=%d (expected %d), test=%d (expected %d)",
                len(s_gmm_val), len(splits.y_val),
                len(s_gmm_test), len(splits.y_test),
            )
    except FileNotFoundError:
        _logger.info("GMM scores not found — skipping GMM in comparison table.")

    # Combined metrics table
    combined_metrics = pd.concat(all_eval_dfs)
    save_table(combined_metrics, "baseline_metrics")

    # Timing table
    timing_df = pd.DataFrame(timing_info)
    save_table(timing_df, "baseline_timing")

    # ══════════════════════════════════════════════════════════════════════
    # 7. Diagnostics
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  DIAGNOSTICS\n" + "=" * 50)

    # Score histograms (validation)
    fig_hist = plot_score_histograms(all_scores_val, splits.y_val, "Val")
    save_fig(fig_hist, "baseline_score_histograms")
    plt.close(fig_hist)

    # Rank correlation matrix
    fig_corr, corr_df = plot_rank_correlation(all_scores_val)
    save_fig(fig_corr, "baseline_rank_correlation")
    save_table(corr_df, "baseline_rank_correlation")
    plt.close(fig_corr)

    # Top-20 analysis for each trained model
    for name in ["pca", "iforest", "ocsvm", "lof"]:
        top_df = analyse_top_k(all_scores_val[name], splits.y_val, name, k=20)
        save_table(top_df, f"baseline_top20_{name}")
        n_tp = (top_df["is_fraud"] == 1).sum()
        n_fp = (top_df["is_fraud"] == 0).sum()
        _logger.info(
            "Top-20 %s: %d caught frauds, %d false positives", name, n_tp, n_fp,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 8. Score direction verification
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "=" * 50 + "\n  SCORE DIRECTION VERIFICATION\n" + "=" * 50)

    for name in model_names:
        s = all_scores_val[name]
        fraud_mean = s[splits.y_val == 1].mean()
        legit_mean = s[splits.y_val == 0].mean()
        direction_ok = fraud_mean > legit_mean
        _logger.info(
            "%s: fraud_mean=%.4f, legit_mean=%.4f, direction_ok=%s",
            name, fraud_mean, legit_mean, direction_ok,
        )
        if not direction_ok and name not in ("random", "amount"):
            _logger.warning(
                "⚠ %s: fraud scores are NOT higher on average — check score convention!",
                name,
            )

    # Verify score lengths
    for name in model_names:
        n_val = len(all_scores_val[name])
        n_test = len(all_scores_test[name])
        assert n_val == len(splits.y_val), f"{name} val length mismatch: {n_val} vs {len(splits.y_val)}"
        assert n_test == len(splits.y_test), f"{name} test length mismatch: {n_test} vs {len(splits.y_test)}"
        _logger.info("%s: val_n=%d, test_n=%d ✓", name, n_val, n_test)

    _logger.info("=" * 70)
    _logger.info("CLASSICAL BASELINES PIPELINE COMPLETE")
    _logger.info("=" * 70)

    return {
        "splits": splits,
        "all_scores_val": all_scores_val,
        "all_scores_test": all_scores_test,
        "all_grid_tables": all_grid_tables,
        "all_metrics": all_metrics,
        "combined_metrics": combined_metrics,
        "timing_df": timing_df,
        "pca_results": pca_results,
        "iforest_results": iforest_results,
        "ocsvm_results": ocsvm_results,
        "lof_results": lof_results,
    }


run_baseline_stage = run_all_baselines

if __name__ == "__main__":
    run_all_baselines()
