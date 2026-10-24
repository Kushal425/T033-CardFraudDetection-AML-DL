"""
src/evaluation.py
=================
Phase 1 Evaluation Toolkit — Krishna Pramod Dubale (Task 04).

Extends the shared ``src.metrics`` module with:
- Bootstrap confidence intervals (stratified, reproducible).
- Paired-bootstrap significance tests between two models.
- Threshold policies (FPR-budget, review-capacity, max-F1) — always
  **chosen on validation, frozen, applied unchanged to test**.
- Cost curves under assumed cost ratios (missed-fraud = r × false-alarm).
- Multi-model comparison table with CIs and significance markers.
- Temporal-stability checks (PR-AUC across time blocks).
- Calibration mapping (Platt / isotonic) with reliability diagrams.
- Publication-quality plots (PR curves, ROC log-x, score histograms, etc.).

**Important**: This module never uses test labels for threshold or model
selection. All tuning uses validation data only.

Score convention (same as ``src.metrics``):
    **Higher score == more anomalous (more likely fraud).**

Dependencies:
    numpy, pandas, scipy, sklearn, matplotlib, seaborn
    src.metrics (shared, read-only)
    src.utils   (shared, read-only)
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from src.metrics import (
    base_rate,
    confusion_at_threshold,
    f1_at_threshold,
    pr_auc,
    precision_at_fraction,
    precision_at_k,
    recall_at_fpr,
    recall_at_k,
    roc_auc,
    threshold_at_fpr,
)
from src.utils import get_logger, save_fig, save_table

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# A. Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    metric_fn: Callable,
    y: np.ndarray,
    s: np.ndarray,
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    **metric_kwargs,
) -> dict:
    """
    Stratified bootstrap 95 % confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn : callable
        ``metric_fn(y, s, **metric_kwargs) -> float``.
        Must accept arrays of ground-truth labels and scores.
    y : array-like of int
        Binary ground-truth labels (1 = fraud).
    s : array-like of float
        Anomaly scores (higher = more anomalous).
    n : int
        Number of bootstrap resamples (default 2000).
    seed : int
        Random seed for reproducibility.
    alpha : float
        Significance level (default 0.05 for 95 % CI).
    **metric_kwargs
        Extra keyword arguments forwarded to *metric_fn* (e.g. ``fpr=0.005``).

    Returns
    -------
    dict
        ``{"point": float, "ci_lo": float, "ci_hi": float,
          "boot_values": ndarray}``

    Notes
    -----
    Stratified resampling: fraud and legitimate indices are resampled
    **separately** so each bootstrap replicate preserves the original class
    ratio.  This is critical under extreme imbalance (0.17 % fraud) because
    naive resampling can produce replicates with zero fraud rows, making some
    metrics undefined and widening the CI unnecessarily.

    The CI is wide on the test set because there are very few frauds
    (e.g. 75 in test): each bootstrap replicate samples ~75 fraud rows with
    replacement, leading to high variance.  We report n_fraud alongside
    all CIs so readers can judge the precision of the estimate.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)

    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    n_fraud = len(fraud_idx)
    n_legit = len(legit_idx)

    point = float(metric_fn(y, s, **metric_kwargs))

    s_fraud = s[fraud_idx]
    s_legit = s[legit_idx]
    y_boot = np.concatenate([np.ones(n_fraud, dtype=int), np.zeros(n_legit, dtype=int)])

    boot_values = np.empty(n, dtype=float)
    for i in range(n):
        # Stratified resampling: resample fraud and legit subsets separately
        f_sample = rng.choice(n_fraud, size=n_fraud, replace=True)
        l_sample = rng.choice(n_legit, size=n_legit, replace=True)
        s_boot = np.concatenate([s_fraud[f_sample], s_legit[l_sample]])
        try:
            boot_values[i] = float(metric_fn(y_boot, s_boot, **metric_kwargs))
        except Exception:
            boot_values[i] = np.nan

    # Remove NaNs if any
    valid = boot_values[~np.isnan(boot_values)]
    if len(valid) == 0:
        warnings.warn("All bootstrap replicates returned NaN.")
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan,
                "boot_values": boot_values}

    ci_lo = float(np.percentile(valid, 100 * alpha / 2))
    ci_hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))

    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "boot_values": boot_values,
    }


def paired_bootstrap_diff(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    metric_fn: Callable,
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    **metric_kwargs,
) -> dict:
    """
    CI of ``metric(A) − metric(B)`` on the **same** bootstrap resamples.

    The difference is "statistically significant at (1−alpha)" if the CI
    excludes zero.

    Parameters
    ----------
    y : array-like of int
        Shared ground-truth labels.
    s_a, s_b : array-like of float
        Scores from model A and model B (same length as *y*).
    metric_fn : callable
        ``metric_fn(y, s, **metric_kwargs) -> float``.
    n : int
        Number of bootstrap resamples.
    seed : int
        Random seed.
    alpha : float
        Significance level.
    **metric_kwargs
        Extra keyword arguments forwarded to *metric_fn*.

    Returns
    -------
    dict
        ``{"diff_point": float, "ci_lo": float, "ci_hi": float,
          "significant": bool, "boot_diffs": ndarray}``
    """
    y = np.asarray(y, dtype=int)
    s_a = np.asarray(s_a, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    rng = np.random.default_rng(seed)

    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    n_fraud = len(fraud_idx)
    n_legit = len(legit_idx)

    point_a = float(metric_fn(y, s_a, **metric_kwargs))
    point_b = float(metric_fn(y, s_b, **metric_kwargs))
    diff_point = point_a - point_b

    sa_fraud, sa_legit = s_a[fraud_idx], s_a[legit_idx]
    sb_fraud, sb_legit = s_b[fraud_idx], s_b[legit_idx]
    y_boot = np.concatenate([np.ones(n_fraud, dtype=int), np.zeros(n_legit, dtype=int)])

    boot_diffs = np.empty(n, dtype=float)
    for i in range(n):
        f_sample = rng.choice(n_fraud, size=n_fraud, replace=True)
        l_sample = rng.choice(n_legit, size=n_legit, replace=True)
        sa_boot = np.concatenate([sa_fraud[f_sample], sa_legit[l_sample]])
        sb_boot = np.concatenate([sb_fraud[f_sample], sb_legit[l_sample]])
        try:
            val_a = float(metric_fn(y_boot, sa_boot, **metric_kwargs))
            val_b = float(metric_fn(y_boot, sb_boot, **metric_kwargs))
            boot_diffs[i] = val_a - val_b
        except Exception:
            boot_diffs[i] = np.nan

    valid = boot_diffs[~np.isnan(boot_diffs)]
    ci_lo = float(np.percentile(valid, 100 * alpha / 2))
    ci_hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    significant = (ci_lo > 0) or (ci_hi < 0)

    return {
        "diff_point": diff_point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "significant": significant,
        "boot_diffs": boot_diffs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# B. Threshold Policies  (always VAL → frozen → applied to TEST)
# ═══════════════════════════════════════════════════════════════════════════════


def threshold_policies_fpr(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_eval: np.ndarray,
    s_eval: np.ndarray,
    fpr_budgets: Sequence[float] = (0.001, 0.005, 0.01),
) -> pd.DataFrame:
    """
    FPR-budget threshold policies.

    For each target FPR, derive the threshold from the *validation*
    legitimate-score quantile, then apply it to the evaluation split.

    Parameters
    ----------
    y_val, s_val : arrays
        Validation labels and scores (used to derive thresholds).
    y_eval, s_eval : arrays
        Evaluation labels and scores (val or test) to measure performance.
    fpr_budgets : sequence of float
        FPR budgets, e.g. (0.001, 0.005, 0.01) for 0.1 %, 0.5 %, 1 %.

    Returns
    -------
    pd.DataFrame
        One row per FPR budget with columns: threshold, n_alerts, TP, FP,
        FN, precision, recall, F1, alerts_per_10k.
    """
    y_val = np.asarray(y_val)
    s_val = np.asarray(s_val)
    y_eval = np.asarray(y_eval)
    s_eval = np.asarray(s_eval)

    rows = []
    for fpr in fpr_budgets:
        thresh = threshold_at_fpr(y_val, s_val, fpr)
        cm = confusion_at_threshold(y_eval, s_eval, thresh)
        tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
        n_alerts = tp + fp
        prec = tp / max(1, n_alerts)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        alerts_per_10k = n_alerts / len(y_eval) * 10_000
        rows.append({
            "policy": f"FPR≤{fpr*100:.1f}%",
            "fpr_budget": fpr,
            "threshold": thresh,
            "n_alerts": n_alerts,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": prec,
            "recall": rec,
            "F1": f1,
            "alerts_per_10k": alerts_per_10k,
        })
    return pd.DataFrame(rows)


def threshold_policies_capacity(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_eval: np.ndarray,
    s_eval: np.ndarray,
    fractions: Sequence[float] = (0.001, 0.005, 0.01),
    top_ks: Sequence[int] = (50, 100, 200),
) -> pd.DataFrame:
    """
    Review-capacity threshold policies: top fraction and top-K.

    The threshold is the minimum score among the top-K (or top-frac)
    transactions in the **validation** set, then applied to the evaluation
    split.

    Parameters
    ----------
    y_val, s_val : arrays
        Validation labels and scores.
    y_eval, s_eval : arrays
        Evaluation split labels and scores.
    fractions : sequence of float
        Fraction of transactions to review.
    top_ks : sequence of int
        Absolute number of transactions to review.

    Returns
    -------
    pd.DataFrame
        One row per policy with threshold, n_alerts, TP, FP, FN,
        precision, recall, F1, alerts_per_10k.
    """
    y_val = np.asarray(y_val)
    s_val = np.asarray(s_val)
    y_eval = np.asarray(y_eval)
    s_eval = np.asarray(s_eval)

    rows = []

    # Fraction-based capacity
    for frac in fractions:
        k_val = max(1, int(np.round(len(y_val) * frac)))
        sorted_val = np.sort(s_val)[::-1]
        thresh = sorted_val[min(k_val - 1, len(sorted_val) - 1)]

        cm = confusion_at_threshold(y_eval, s_eval, thresh)
        tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
        n_alerts = tp + fp
        prec = tp / max(1, n_alerts)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        alerts_per_10k = n_alerts / len(y_eval) * 10_000
        rows.append({
            "policy": f"top-{frac*100:.1f}%",
            "review_param": frac,
            "threshold": thresh,
            "n_alerts": n_alerts,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": prec,
            "recall": rec,
            "F1": f1,
            "alerts_per_10k": alerts_per_10k,
        })

    # Absolute top-K capacity
    for k in top_ks:
        k_actual = min(k, len(s_val))
        sorted_val = np.sort(s_val)[::-1]
        thresh = sorted_val[k_actual - 1]

        cm = confusion_at_threshold(y_eval, s_eval, thresh)
        tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
        n_alerts = tp + fp
        prec = tp / max(1, n_alerts)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        alerts_per_10k = n_alerts / len(y_eval) * 10_000
        rows.append({
            "policy": f"top-K={k}",
            "review_param": k,
            "threshold": thresh,
            "n_alerts": n_alerts,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": prec,
            "recall": rec,
            "F1": f1,
            "alerts_per_10k": alerts_per_10k,
        })

    return pd.DataFrame(rows)


def threshold_max_f1_val(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_eval: np.ndarray,
    s_eval: np.ndarray,
    n_thresholds: int = 1000,
) -> dict:
    """
    Find the threshold that maximises F1 on the **validation** set, then
    apply it (frozen) to the evaluation split.

    Parameters
    ----------
    y_val, s_val : arrays
        Validation labels and scores.
    y_eval, s_eval : arrays
        Evaluation split labels and scores.
    n_thresholds : int
        Number of candidate thresholds to sweep.

    Returns
    -------
    dict
        ``threshold``, ``val_f1``, plus confusion/metric dict for eval split.
    """
    y_val = np.asarray(y_val)
    s_val = np.asarray(s_val)
    y_eval = np.asarray(y_eval)
    s_eval = np.asarray(s_eval)

    candidates = np.linspace(s_val.min(), s_val.max(), n_thresholds)
    best_f1 = -1.0
    best_thresh = candidates[0]
    for t in candidates:
        f1 = f1_at_threshold(y_val, s_val, t)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Apply frozen threshold to eval
    cm = confusion_at_threshold(y_eval, s_eval, best_thresh)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    n_alerts = tp + fp
    prec = tp / max(1, n_alerts)
    rec = tp / max(1, tp + fn)
    f1_eval = 2 * prec * rec / max(1e-12, prec + rec)

    return {
        "threshold": best_thresh,
        "val_f1": best_f1,
        "eval_f1": f1_eval,
        "eval_precision": prec,
        "eval_recall": rec,
        "TP": tp, "FP": fp, "FN": fn,
        "n_alerts": n_alerts,
        "alerts_per_10k": n_alerts / len(y_eval) * 10_000,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# C. Cost Curve
# ═══════════════════════════════════════════════════════════════════════════════


def cost_curve(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_eval: np.ndarray,
    s_eval: np.ndarray,
    cost_ratios: Sequence[float] = (10, 25, 50, 100, 250, 500),
    n_thresholds: int = 500,
) -> pd.DataFrame:
    """
    Sweep cost ratios and thresholds to find the optimal operating point.

    **ASSUMPTION**: The cost of a missed fraud is *r* times the cost of a
    false alarm.  Both costs are notional and problem-dependent; the exact
    values are unknowable without domain-specific loss data.  We provide
    this analysis to show *how* the optimal threshold shifts with different
    cost assumptions, not to prescribe a single "correct" ratio.

    Cost per transaction = (FN / N) × r + (FP / N) × 1

    Parameters
    ----------
    y_val, s_val : arrays
        Validation labels and scores (threshold derived here).
    y_eval, s_eval : arrays
        Evaluation labels and scores.
    cost_ratios : sequence of float
        Values of r (missed-fraud cost / false-alarm cost).
    n_thresholds : int
        Number of thresholds to sweep on validation.

    Returns
    -------
    pd.DataFrame
        Columns: cost_ratio, optimal_threshold, min_expected_cost,
        TP, FP, FN, TN (at the optimal threshold applied to eval).
    """
    y_val = np.asarray(y_val)
    s_val = np.asarray(s_val)
    y_eval = np.asarray(y_eval)
    s_eval = np.asarray(s_eval)

    candidates = np.linspace(s_val.min(), s_val.max(), n_thresholds)
    n_val = len(y_val)
    n_eval = len(y_eval)

    rows = []
    for r in cost_ratios:
        # Find optimal threshold on VAL
        best_cost = np.inf
        best_thresh = candidates[0]
        for t in candidates:
            cm = confusion_at_threshold(y_val, s_val, t)
            cost = (cm["FN"] / n_val) * r + (cm["FP"] / n_val) * 1.0
            if cost < best_cost:
                best_cost = cost
                best_thresh = t

        # Apply to eval (frozen threshold)
        cm_eval = confusion_at_threshold(y_eval, s_eval, best_thresh)
        eval_cost = (cm_eval["FN"] / n_eval) * r + (cm_eval["FP"] / n_eval) * 1.0
        rows.append({
            "cost_ratio": r,
            "optimal_threshold": best_thresh,
            "val_expected_cost": best_cost,
            "eval_expected_cost": eval_cost,
            "TP": cm_eval["TP"],
            "FP": cm_eval["FP"],
            "FN": cm_eval["FN"],
            "TN": cm_eval["TN"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Calibration (optional — scores are NOT probabilities)
# ═══════════════════════════════════════════════════════════════════════════════


def calibrate_scores_platt(
    s_val: np.ndarray,
    y_val: np.ndarray,
    s_eval: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
    """
    Map raw anomaly scores to pseudo-probabilities via Platt scaling (logistic
    regression on val), then apply to eval.

    **Caveat**: The resulting probabilities are only as good as the
    calibration data.  With very few validation frauds (n≈57), Platt
    scaling may be unreliable.

    Returns
    -------
    tuple of (calibrated_probs, fitted_model)
    """
    s_val = np.asarray(s_val).reshape(-1, 1)
    s_eval = np.asarray(s_eval).reshape(-1, 1)
    y_val = np.asarray(y_val)

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(s_val, y_val)
    probs = lr.predict_proba(s_eval)[:, 1]
    return probs, lr


def calibrate_scores_isotonic(
    s_val: np.ndarray,
    y_val: np.ndarray,
    s_eval: np.ndarray,
) -> tuple[np.ndarray, IsotonicRegression]:
    """
    Map raw anomaly scores to pseudo-probabilities via isotonic regression
    fitted on val.

    **Caveat**: Isotonic regression is non-parametric and can overfit with
    very few positive examples.

    Returns
    -------
    tuple of (calibrated_probs, fitted_model)
    """
    s_val = np.asarray(s_val, dtype=float)
    s_eval = np.asarray(s_eval, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s_val, y_val)
    probs = iso.predict(s_eval)
    return probs, iso


# ═══════════════════════════════════════════════════════════════════════════════
# E. Plotting Utilities
# ═══════════════════════════════════════════════════════════════════════════════

# Consistent colour palette for models
MODEL_COLOURS = {
    "gmm": "#2196F3",
    "iforest": "#FF9800",
    "lof": "#4CAF50",
    "ocsvm": "#E91E63",
    "pca": "#9C27B0",
    "amount": "#795548",
    "random": "#9E9E9E",
}


def _get_colour(name: str) -> str:
    return MODEL_COLOURS.get(name.lower(), "#000000")


def plot_pr_curves(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    title: str = "Precision–Recall Curves",
    save_name: str | None = None,
) -> plt.Figure:
    """
    PR curves for multiple models on the same axes, with base-rate line.

    Parameters
    ----------
    models : dict
        ``{model_name: (y_split, scores_split)}``.
    y : array-like
        Ground-truth labels (same for all models).
    title : str
        Plot title.
    save_name : str, optional
        If given, save via ``save_fig``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(8, 6))
    br = base_rate(y)

    for name, (y_m, s_m) in models.items():
        prec_arr, rec_arr, _ = precision_recall_curve(np.asarray(y_m), np.asarray(s_m))
        ap = pr_auc(y_m, s_m)
        ax.plot(rec_arr, prec_arr, label=f"{name} (AP={ap:.4f})",
                color=_get_colour(name), linewidth=1.5)

    ax.axhline(y=br, color="grey", linestyle="--", linewidth=1,
               label=f"base rate = {br:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_roc_logx(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curve (log FPR axis)",
    save_name: str | None = None,
) -> plt.Figure:
    """
    ROC curves with logarithmic x-axis (FPR) for all models.
    """
    from sklearn.metrics import roc_curve as sk_roc_curve

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_m, s_m) in models.items():
        fpr_arr, tpr_arr, _ = sk_roc_curve(np.asarray(y_m), np.asarray(s_m))
        auc_val = roc_auc(y_m, s_m)
        # Filter out FPR = 0 for log scale
        mask = fpr_arr > 0
        ax.plot(fpr_arr[mask], tpr_arr[mask],
                label=f"{name} (AUC={auc_val:.4f})",
                color=_get_colour(name), linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("False Positive Rate (log scale)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_recall_vs_fpr(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    fpr_budgets: Sequence[float] = (0.001, 0.005, 0.01),
    title: str = "Recall at Fixed FPR Budgets",
    save_name: str | None = None,
) -> plt.Figure:
    """
    Bar chart of recall at each FPR budget for all models.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = list(models.keys())
    n_models = len(model_names)
    n_budgets = len(fpr_budgets)
    x = np.arange(n_budgets)
    width = 0.8 / max(1, n_models)

    for i, name in enumerate(model_names):
        y_m, s_m = models[name]
        recalls = [recall_at_fpr(y_m, s_m, fpr) for fpr in fpr_budgets]
        ax.bar(x + i * width, recalls, width, label=name,
               color=_get_colour(name), alpha=0.8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"FPR≤{b*100:.1f}%" for b in fpr_budgets])
    ax.set_ylabel("Recall")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_precision_at_k_curve(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    max_k: int = 500,
    step: int = 10,
    title: str = "Precision@K Curve",
    save_name: str | None = None,
) -> plt.Figure:
    """
    Precision@K as a function of K for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ks = list(range(step, max_k + 1, step))

    for name, (y_m, s_m) in models.items():
        precs = [precision_at_k(y_m, s_m, k) for k in ks]
        ax.plot(ks, precs, label=name, color=_get_colour(name), linewidth=1.5)

    ax.set_xlabel("K (review capacity)")
    ax.set_ylabel("Precision@K")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_score_histograms(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    title_prefix: str = "Score Distribution",
    save_name: str | None = None,
) -> plt.Figure:
    """
    Score histograms (legitimate vs fraud) for each model, with log y-axis.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.flatten()

    for i, (name, (y_m, s_m)) in enumerate(models.items()):
        y_m = np.asarray(y_m)
        s_m = np.asarray(s_m)
        ax = axes[i]
        ax.hist(s_m[y_m == 0], bins=80, alpha=0.6, label="Legit",
                color="#2196F3", density=True)
        ax.hist(s_m[y_m == 1], bins=80, alpha=0.7, label="Fraud",
                color="#F44336", density=True)
        ax.set_yscale("log")
        ax.set_title(f"{title_prefix}: {name}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density (log)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_cost_curve(
    cost_df: pd.DataFrame,
    title: str = "Expected Cost vs Cost Ratio",
    save_name: str | None = None,
) -> plt.Figure:
    """
    Plot expected cost at the optimal threshold for each cost ratio.

    Parameters
    ----------
    cost_df : pd.DataFrame
        Output of ``cost_curve()``.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_df["cost_ratio"], cost_df["eval_expected_cost"],
            marker="o", linewidth=2, color="#E91E63")
    ax.set_xlabel("Cost ratio r  (missed fraud / false alarm)")
    ax.set_ylabel("Expected cost per transaction")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Annotate optimal thresholds
    for _, row in cost_df.iterrows():
        ax.annotate(f"θ={row['optimal_threshold']:.3f}",
                    (row["cost_ratio"], row["eval_expected_cost"]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center")

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_reliability_diagram(
    y: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram (Calibration)",
    save_name: str | None = None,
) -> plt.Figure:
    """
    Reliability diagram for calibrated pseudo-probabilities.

    **Caveat**: Anomaly scores are not probabilities by default.  This plot
    is meaningful only after applying Platt/isotonic calibration on validation
    data.  With very few frauds, the calibration curve may be noisy.
    """
    y = np.asarray(y)
    probs = np.asarray(probs)
    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        fraction_pos, mean_predicted = calibration_curve(y, probs, n_bins=n_bins)
        ax.plot(mean_predicted, fraction_pos, marker="s", linewidth=2,
                label="Model", color="#2196F3")
    except ValueError:
        ax.text(0.5, 0.5, "Insufficient data for calibration curve",
                ha="center", va="center", transform=ax.transAxes)

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# F. Model Comparison Table
# ═══════════════════════════════════════════════════════════════════════════════


def compare_models(
    model_scores: dict[str, dict],
    y_val: np.ndarray,
    y_test: np.ndarray,
    fpr_budgets: Sequence[float] = (0.001, 0.005, 0.01),
    review_fractions: Sequence[float] = (0.001, 0.005, 0.01),
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Master comparison table with CIs for all models.

    Parameters
    ----------
    model_scores : dict
        ``{model_name: {"val": s_val, "test": s_test,
                        "train_time_s": float, "inference_time_s": float}}``
    y_val, y_test : arrays
        Ground-truth labels for val and test splits.
    fpr_budgets : sequence of float
    review_fractions : sequence of float
    n_boot : int
        Bootstrap resamples.
    seed : int

    Returns
    -------
    pd.DataFrame
        Master table saved as ``eval_master_phase1.csv``.
    """
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    rows = []
    for name, info in model_scores.items():
        s_val = np.asarray(info["val"])
        s_test = np.asarray(info["test"])

        row = {"model": name}

        # PR-AUC with CI (test)
        ci = bootstrap_ci(pr_auc, y_test, s_test, n=n_boot, seed=seed)
        row["PR-AUC_test"] = ci["point"]
        row["PR-AUC_test_CI"] = f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]"

        # PR-AUC (val)
        row["PR-AUC_val"] = pr_auc(y_val, s_val)

        # ROC-AUC with CI (test)
        ci_roc = bootstrap_ci(roc_auc, y_test, s_test, n=n_boot, seed=seed)
        row["ROC-AUC_test"] = ci_roc["point"]
        row["ROC-AUC_test_CI"] = f"[{ci_roc['ci_lo']:.4f}, {ci_roc['ci_hi']:.4f}]"

        # Recall@FPR with CI (test)
        for fpr in fpr_budgets:
            ci_fpr = bootstrap_ci(
                recall_at_fpr, y_test, s_test,
                n=n_boot, seed=seed, fpr=fpr,
            )
            key = f"recall@FPR={fpr*100:.1f}%_test"
            row[key] = ci_fpr["point"]
            row[key + "_CI"] = f"[{ci_fpr['ci_lo']:.4f}, {ci_fpr['ci_hi']:.4f}]"

        # Precision@capacity (test)
        for frac in review_fractions:
            key = f"prec@cap={frac*100:.1f}%_test"
            row[key] = precision_at_fraction(y_test, s_test, frac)

        # Timing
        row["train_time_s"] = info.get("train_time_s", np.nan)
        row["inference_time_s"] = info.get("inference_time_s", np.nan)

        # Fraud counts for context
        row["n_fraud_test"] = int(y_test.sum())
        row["n_total_test"] = len(y_test)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    # Rank by PR-AUC test (descending)
    df = df.sort_values("PR-AUC_test", ascending=False)

    # Paired bootstrap significance vs the best model
    best_name = df.index[0]
    s_best_test = np.asarray(model_scores[best_name]["test"])
    sig_marks = {}
    for name in df.index:
        if name == best_name:
            sig_marks[name] = "best"
            continue
        s_other = np.asarray(model_scores[name]["test"])
        result = paired_bootstrap_diff(
            y_test, s_best_test, s_other, pr_auc, n=n_boot, seed=seed,
        )
        if result["significant"]:
            sig_marks[name] = "sig. worse"
        else:
            sig_marks[name] = "indistinguishable"

    df["vs_best"] = pd.Series(sig_marks)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# G. Temporal Stability Check
# ═══════════════════════════════════════════════════════════════════════════════


def temporal_stability(
    y: np.ndarray,
    s: np.ndarray,
    time_col: np.ndarray,
    n_blocks: int = 5,
    metric_fn: Callable = pr_auc,
    min_fraud_per_block: int = 5,
) -> pd.DataFrame:
    """
    Compute metric on consecutive time blocks to check for temporal drift.

    Parameters
    ----------
    y : array-like
        Ground-truth labels.
    s : array-like
        Scores.
    time_col : array-like
        Timestamps or time feature for ordering.
    n_blocks : int
        Number of consecutive time blocks to split into.
    metric_fn : callable
        Metric function (default ``pr_auc``).
    min_fraud_per_block : int
        Minimum frauds required in a block to compute the metric.

    Returns
    -------
    pd.DataFrame
        block_id, time_min, time_max, n_total, n_fraud, metric_value.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    time_col = np.asarray(time_col)

    # Sort by time
    order = np.argsort(time_col)
    y = y[order]
    s = s[order]
    time_col = time_col[order]

    block_size = len(y) // n_blocks
    rows = []
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else len(y)
        y_b = y[start:end]
        s_b = s[start:end]
        t_b = time_col[start:end]

        n_fraud = int(y_b.sum())
        if n_fraud < min_fraud_per_block:
            metric_val = np.nan
            note = f"<{min_fraud_per_block} frauds; metric unreliable"
        else:
            metric_val = float(metric_fn(y_b, s_b))
            note = ""

        rows.append({
            "block": b + 1,
            "time_min": float(t_b.min()),
            "time_max": float(t_b.max()),
            "n_total": len(y_b),
            "n_fraud": n_fraud,
            "metric": metric_val,
            "note": note,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# H. Convenience: Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_full_evaluation(
    model_scores: dict[str, dict],
    y_val: np.ndarray,
    s_val_dict: dict[str, np.ndarray] | None = None,
    y_test: np.ndarray | None = None,
    time_test: np.ndarray | None = None,
    cfg: dict | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Run the complete evaluation pipeline and return all results.

    This is the entry point used by the evaluation notebook.  It:
    1. Computes the master comparison table with CIs.
    2. Generates all threshold policy tables.
    3. Computes cost curves.
    4. Checks temporal stability.
    5. Generates and saves all plots.

    Parameters
    ----------
    model_scores : dict
        ``{model_name: {"val": s_val, "test": s_test}}``
    y_val : array
        Validation labels.
    y_test : array
        Test labels.
    time_test : array, optional
        Time column for temporal stability check.
    cfg : dict, optional
        Config dict (uses defaults from config.yaml if None).
    n_boot : int
    seed : int

    Returns
    -------
    dict
        Keys: ``master_table``, ``threshold_fpr``, ``threshold_capacity``,
        ``max_f1``, ``cost_curve``, ``temporal``.
    """
    from src.config import CFG
    if cfg is None:
        cfg = CFG

    y_val = np.asarray(y_val)
    if y_test is not None:
        y_test = np.asarray(y_test)

    fpr_budgets = cfg["eval"]["fpr_budgets"]
    review_fracs = cfg["eval"]["review_fractions"]
    top_ks = cfg["eval"]["top_k"]

    results = {}

    # ── Master table ──────────────────────────────────────────────────────
    if y_test is not None:
        logger.info("Computing master comparison table with %d bootstrap resamples...", n_boot)
        master = compare_models(
            model_scores, y_val, y_test,
            fpr_budgets=fpr_budgets,
            review_fractions=review_fracs,
            n_boot=n_boot, seed=seed,
        )
        save_table(master, "eval_master_phase1")
        results["master_table"] = master
        logger.info("Master table:\n%s", master.to_string())

    # ── Threshold policies (per model) ────────────────────────────────────
    all_thresh_fpr = {}
    all_thresh_cap = {}
    all_maxf1 = {}
    all_cost = {}

    for name, info in model_scores.items():
        s_val = np.asarray(info["val"])
        s_test_m = np.asarray(info["test"]) if "test" in info else None

        if y_test is not None and s_test_m is not None:
            # FPR-budget thresholds
            tf = threshold_policies_fpr(y_val, s_val, y_test, s_test_m, fpr_budgets)
            tf["model"] = name
            tf["split"] = "test"
            all_thresh_fpr[name] = tf

            # Capacity thresholds
            tc = threshold_policies_capacity(
                y_val, s_val, y_test, s_test_m, review_fracs, top_ks)
            tc["model"] = name
            tc["split"] = "test"
            all_thresh_cap[name] = tc

            # Max-F1
            mf = threshold_max_f1_val(y_val, s_val, y_test, s_test_m)
            mf["model"] = name
            all_maxf1[name] = mf

            # Cost curve
            cc = cost_curve(y_val, s_val, y_test, s_test_m)
            cc["model"] = name
            all_cost[name] = cc

    if all_thresh_fpr:
        df_fpr = pd.concat(all_thresh_fpr.values(), ignore_index=True)
        save_table(df_fpr, "eval_threshold_fpr")
        results["threshold_fpr"] = df_fpr

    if all_thresh_cap:
        df_cap = pd.concat(all_thresh_cap.values(), ignore_index=True)
        save_table(df_cap, "eval_threshold_capacity")
        results["threshold_capacity"] = df_cap

    if all_maxf1:
        df_maxf1 = pd.DataFrame(all_maxf1).T
        save_table(df_maxf1, "eval_max_f1")
        results["max_f1"] = df_maxf1

    if all_cost:
        df_cost = pd.concat(all_cost.values(), ignore_index=True)
        save_table(df_cost, "eval_cost_curve")
        results["cost_curve"] = df_cost

    # ── Plots ─────────────────────────────────────────────────────────────
    if y_test is not None:
        test_models = {
            n: (y_test, np.asarray(info["test"]))
            for n, info in model_scores.items()
            if "test" in info
        }

        plot_pr_curves(test_models, y_test,
                       title="PR Curves — Test Set", save_name="eval_pr_curves")
        plot_roc_logx(test_models,
                      title="ROC Curves (log FPR) — Test Set",
                      save_name="eval_roc_logx")
        plot_recall_vs_fpr(test_models, fpr_budgets,
                           title="Recall at FPR Budgets — Test Set",
                           save_name="eval_recall_at_fpr")
        plot_precision_at_k_curve(test_models,
                                  title="Precision@K — Test Set",
                                  save_name="eval_precision_at_k")
        plot_score_histograms(test_models,
                              title_prefix="Score Distribution (Test)",
                              save_name="eval_score_histograms")

        # Cost curve plot (for best model)
        if all_cost:
            best_model = list(model_scores.keys())[0]
            if best_model in all_cost:
                plot_cost_curve(
                    all_cost[best_model],
                    title=f"Cost Curve — {best_model}",
                    save_name="eval_cost_curve",
                )

    # ── Temporal stability ────────────────────────────────────────────────
    if y_test is not None and time_test is not None:
        temp_results = {}
        for name, info in model_scores.items():
            if "test" in info:
                ts = temporal_stability(
                    y_test, np.asarray(info["test"]), time_test,
                    n_blocks=5,
                )
                ts["model"] = name
                temp_results[name] = ts
        if temp_results:
            df_temp = pd.concat(temp_results.values(), ignore_index=True)
            save_table(df_temp, "eval_temporal_stability")
            results["temporal"] = df_temp

    plt.close("all")
    return results
