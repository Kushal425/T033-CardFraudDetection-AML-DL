"""
src/metrics.py
==============
Single source of truth for all anomaly-detection evaluation metrics.
Team T033 — Card Fraud Triage (Phase 1: ML only).

Score convention (enforced everywhere): **higher score == more anomalous
(more likely fraud)**.

Public API
----------
- ``pr_auc(y, s)``                          → average precision (PR-AUC)
- ``roc_auc(y, s)``                         → ROC-AUC
- ``recall_at_fpr(y, s, fpr)``             → highest recall s.t. FPR ≤ fpr
- ``threshold_at_fpr(y_val, s_val, fpr)``  → threshold from val-legit quantile
- ``precision_at_k(y, s, k)``              → precision among top-k scored
- ``precision_at_fraction(y, s, frac)``    → precision at review-capacity fraction
- ``recall_at_k(y, s, k)``                 → recall among top-k scored
- ``f1_at_threshold(y, s, thresh)``        → F1 at a fixed decision threshold
- ``confusion_at_threshold(y, s, thresh)`` → dict with TP/FP/TN/FN
- ``base_rate(y)``                          → fraud fraction
- ``evaluate_scores(y_val, s_val, y_test, s_test, cfg)``
                                            → tidy DataFrame (metric × split)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


# ═══════════════════════════════════════════════════════════════════════════
# Basic rate
# ═══════════════════════════════════════════════════════════════════════════

def base_rate(y: np.ndarray) -> float:
    """
    Return the fraction of positive (fraud) labels in *y*.

    Parameters
    ----------
    y : array-like of int
        Binary labels: 1 = fraud, 0 = legitimate.

    Returns
    -------
    float
        Fraud base rate in [0, 1].

    Notes
    -----
    The base rate in the ULB dataset is ≈ 0.00172 (0.172 %).  Never report
    accuracy as the headline metric because a model that always predicts
    'legitimate' achieves > 99.8 % accuracy.
    """
    y = np.asarray(y)
    return float(y.mean())


# ═══════════════════════════════════════════════════════════════════════════
# Ranking metrics
# ═══════════════════════════════════════════════════════════════════════════

def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """
    Precision-Recall Area Under Curve (average precision).

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels (1 = fraud).
    s : array-like of float
        Anomaly scores; **higher == more anomalous**.

    Returns
    -------
    float
        Average precision in [0, 1].  A perfect ranker returns 1.0; a random
        ranker returns ≈ base_rate(y).

    Notes
    -----
    We use ``sklearn.metrics.average_precision_score`` which computes the
    area under the Precision-Recall curve using the step-function
    interpolation (``average='micro'`` in sklearn terms).  This is the primary
    headline metric for this project because PR-AUC is not inflated by true
    negatives (unlike ROC-AUC) under severe class imbalance.
    """
    return float(average_precision_score(np.asarray(y), np.asarray(s)))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """
    Area Under the Receiver-Operating-Characteristic Curve.

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels.
    s : array-like of float
        Anomaly scores; higher == more anomalous.

    Returns
    -------
    float
        ROC-AUC in [0, 1].

    Notes
    -----
    A perfect ranker returns 1.0; a random ranker returns 0.5.  ROC-AUC is
    reported for completeness but **must not be the headline metric** given
    the severe class imbalance in this dataset.
    """
    return float(roc_auc_score(np.asarray(y), np.asarray(s)))


# ═══════════════════════════════════════════════════════════════════════════
# FPR-anchored metrics
# ═══════════════════════════════════════════════════════════════════════════

def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """
    Return the highest recall achievable while keeping FPR ≤ *fpr*.

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels.
    s : array-like of float
        Anomaly scores; higher == more anomalous.
    fpr : float
        False-positive-rate budget in [0, 1], e.g. 0.001 for 0.1 %.

    Returns
    -------
    float
        Best recall achievable at FPR ≤ fpr.  Returns 0.0 if no operating
        point satisfies the budget.

    Algorithm
    ---------
    1. Compute the full ROC curve (sklearn returns arrays of fpr, tpr, thresholds).
    2. Filter to operating points where fpr_array ≤ budget.
    3. Return max(tpr) at those points.

    Notes
    -----
    The ROC curve includes the trivial endpoints (0, 0) and (1, 1).
    """
    y = np.asarray(y)
    s = np.asarray(s)
    fpr_arr, tpr_arr, _ = roc_curve(y, s)
    mask = fpr_arr <= fpr
    if not mask.any():
        return 0.0
    return float(tpr_arr[mask].max())


def threshold_at_fpr(
    y_val: np.ndarray,
    s_val: np.ndarray,
    fpr: float,
) -> float:
    """
    Derive a decision threshold from **validation** legitimate scores.

    Strategy: use the (1 - fpr) quantile of the val-legit score distribution
    as the threshold.  This means roughly *fpr* fraction of legitimate
    val transactions would be flagged at this threshold.

    Parameters
    ----------
    y_val : array-like of int
        Ground-truth labels for the validation split.
    s_val : array-like of float
        Anomaly scores for the validation split; higher == more anomalous.
    fpr : float
        Target false-positive rate budget, e.g. 0.005 for 0.5 %.

    Returns
    -------
    float
        The score threshold.  Flag a transaction if score ≥ threshold.

    Leakage note
    ------------
    Only the LEGITIMATE rows of the validation set are used to derive this
    threshold.  Fraud labels are used for selection only in the sense that we
    isolate them (exclude them); the threshold value itself is determined
    purely by the legitimate distribution.
    """
    y_val = np.asarray(y_val)
    s_val = np.asarray(s_val)
    legit_scores = s_val[y_val == 0]
    if len(legit_scores) == 0:
        raise ValueError("No legitimate rows in validation set.")
    return float(np.quantile(legit_scores, 1.0 - fpr))


# ═══════════════════════════════════════════════════════════════════════════
# Top-K / review-capacity metrics
# ═══════════════════════════════════════════════════════════════════════════

def _top_k_mask(s: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean mask for the top-*k* highest-scored items."""
    s = np.asarray(s)
    if k >= len(s):
        return np.ones(len(s), dtype=bool)
    # argsort ascending; take the last k indices
    idx = np.argpartition(s, -k)[-k:]
    mask = np.zeros(len(s), dtype=bool)
    mask[idx] = True
    return mask


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """
    Fraction of frauds among the top-*k* highest-scored transactions.

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels.
    s : array-like of float
        Anomaly scores; higher == more anomalous.
    k : int
        Number of top-scored transactions to inspect (review capacity).

    Returns
    -------
    float
        Precision = TP / k.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    mask = _top_k_mask(s, k)
    return float(y[mask].mean())


def precision_at_fraction(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    """
    Precision when flagging the top *frac* fraction of transactions.

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels.
    s : array-like of float
        Anomaly scores; higher == more anomalous.
    frac : float
        Review capacity as a fraction of total transactions (e.g. 0.01 = 1 %).

    Returns
    -------
    float
        Precision within the top-frac flagged set.
    """
    k = max(1, int(np.round(len(y) * frac)))
    return precision_at_k(y, s, k)


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """
    Fraction of ALL frauds that appear in the top-*k* scored transactions.

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels.
    s : array-like of float
        Anomaly scores; higher == more anomalous.
    k : int
        Review budget.

    Returns
    -------
    float
        Recall = TP / (TP + FN).  Returns 1.0 if there are no frauds.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    n_fraud = y.sum()
    if n_fraud == 0:
        return 1.0
    mask = _top_k_mask(s, k)
    return float(y[mask].sum() / n_fraud)


# ═══════════════════════════════════════════════════════════════════════════
# Threshold-based metrics
# ═══════════════════════════════════════════════════════════════════════════

def confusion_at_threshold(
    y: np.ndarray, s: np.ndarray, thresh: float
) -> dict[str, int]:
    """
    Confusion matrix at a fixed decision threshold.

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels.
    s : array-like of float
        Anomaly scores; higher == more anomalous.
    thresh : float
        Flag as fraud if score ≥ thresh.

    Returns
    -------
    dict
        Keys: ``"TP"``, ``"FP"``, ``"TN"``, ``"FN"``.
    """
    y = np.asarray(y)
    pred = (np.asarray(s) >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def f1_at_threshold(y: np.ndarray, s: np.ndarray, thresh: float) -> float:
    """
    F1 score at a fixed decision threshold.

    Parameters
    ----------
    y : array-like of int
        Binary ground-truth labels.
    s : array-like of float
        Anomaly scores; higher == more anomalous.
    thresh : float
        Flag as fraud if score ≥ thresh.

    Returns
    -------
    float
        F1 in [0, 1].  Returns 0.0 if no positives are predicted.
    """
    cm = confusion_at_threshold(y, s, thresh)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Composite evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_scores(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    cfg: dict,
) -> pd.DataFrame:
    """
    Compute the full evaluation table for one model.

    Thresholds are chosen on the **validation** split and **applied unchanged**
    to the test split.  This respects the leakage rules: test labels are used
    only for evaluation, never for model or threshold selection.

    Parameters
    ----------
    y_val : array-like of int
        Validation ground-truth labels.
    s_val : array-like of float
        Validation anomaly scores (higher == more anomalous).
    y_test : array-like of int
        Test ground-truth labels.
    s_test : array-like of float
        Test anomaly scores (higher == more anomalous).
    cfg : dict
        Project config dict (loaded from ``configs/config.yaml``).  Used for
        ``cfg['eval']['fpr_budgets']``, ``cfg['eval']['review_fractions']``,
        and ``cfg['eval']['top_k']``.

    Returns
    -------
    pd.DataFrame
        Index = metric names, columns = ``["val", "test"]``.

    Notes
    -----
    Accuracy is intentionally EXCLUDED from the returned table.  The headline
    metric is PR-AUC.
    """
    y_val = np.asarray(y_val)
    s_val = np.asarray(s_val)
    y_test = np.asarray(y_test)
    s_test = np.asarray(s_test)

    results: dict[str, list] = {}  # metric_name -> [val_value, test_value]

    # ── Global ranking metrics ────────────────────────────────────────────
    results["PR-AUC"] = [pr_auc(y_val, s_val), pr_auc(y_test, s_test)]
    results["ROC-AUC"] = [roc_auc(y_val, s_val), roc_auc(y_test, s_test)]
    results["base_rate"] = [base_rate(y_val), base_rate(y_test)]
    results["n_fraud"] = [int(y_val.sum()), int(y_test.sum())]
    results["n_total"] = [len(y_val), len(y_test)]

    # ── Recall @ FPR budgets ──────────────────────────────────────────────
    for fpr in cfg["eval"]["fpr_budgets"]:
        key = f"recall@FPR={fpr:.3f}"
        thresh = threshold_at_fpr(y_val, s_val, fpr)
        results[key] = [
            recall_at_fpr(y_val, s_val, fpr),
            recall_at_fpr(y_test, s_test, fpr),
        ]
        # Also store the threshold-based F1 applied to test
        cm_v = confusion_at_threshold(y_val, s_val, thresh)
        cm_t = confusion_at_threshold(y_test, s_test, thresh)
        results[f"F1@FPR={fpr:.3f}"] = [
            f1_at_threshold(y_val, s_val, thresh),
            f1_at_threshold(y_test, s_test, thresh),
        ]
        results[f"precision@FPR={fpr:.3f}"] = [
            cm_v["TP"] / max(1, cm_v["TP"] + cm_v["FP"]),
            cm_t["TP"] / max(1, cm_t["TP"] + cm_t["FP"]),
        ]

    # ── Precision @ review capacity fractions ────────────────────────────
    for frac in cfg["eval"]["review_fractions"]:
        key = f"precision@cap={frac:.3f}"
        results[key] = [
            precision_at_fraction(y_val, s_val, frac),
            precision_at_fraction(y_test, s_test, frac),
        ]

    # ── Top-K metrics ─────────────────────────────────────────────────────
    for k in cfg["eval"]["top_k"]:
        results[f"precision@K={k}"] = [
            precision_at_k(y_val, s_val, k),
            precision_at_k(y_test, s_test, k),
        ]
        results[f"recall@K={k}"] = [
            recall_at_k(y_val, s_val, k),
            recall_at_k(y_test, s_test, k),
        ]

    df = pd.DataFrame(results, index=["val", "test"]).T
    return df
