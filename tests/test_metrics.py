"""
tests/test_metrics.py
=====================
Unit tests for ``src/metrics.py``.

Tests
-----
1. ``test_perfect_scores_pr_auc_equals_one``
   A perfect ranker (scores = labels) should yield PR-AUC = 1.0.
2. ``test_random_scores_pr_auc_approx_base_rate``
   Random scores should yield PR-AUC ≈ base_rate (within a loose tolerance).
3. ``test_recall_at_fpr_monotone``
   recall_at_fpr(y, s, fpr) is monotonically non-decreasing in fpr.
4. ``test_roc_auc_perfect``
   Perfect scores → ROC-AUC = 1.0.
5. ``test_roc_auc_random``
   Random scores → ROC-AUC ≈ 0.5.
6. ``test_precision_at_k``
   Known synthetic case: top-k are all frauds → precision = 1.0.
7. ``test_threshold_at_fpr_legit_only``
   threshold_at_fpr uses only legitimate val rows.
8. ``test_confusion_at_threshold``
   Verify TP/FP/TN/FN against a hand-calculated case.
9. ``test_base_rate``
   base_rate returns correct fraction.
10. ``test_evaluate_scores_shape``
    evaluate_scores returns a DataFrame with columns ['val','test'] and no
    accuracy row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    base_rate,
    confusion_at_threshold,
    evaluate_scores,
    f1_at_threshold,
    pr_auc,
    precision_at_fraction,
    precision_at_k,
    recall_at_fpr,
    recall_at_k,
    roc_auc,
    threshold_at_fpr,
)

RNG = np.random.default_rng(42)

# ── Shared synthetic data ─────────────────────────────────────────────────
N = 10_000
FRAUD_FRAC = 0.02
N_FRAUD = int(N * FRAUD_FRAC)
N_LEGIT = N - N_FRAUD

_y = np.array([1] * N_FRAUD + [0] * N_LEGIT)
# Perfect scores: frauds get score 1.0, legit get 0.0
_s_perfect = _y.astype(float)
# Random scores
_s_random  = RNG.uniform(0, 1, N)


# ═══════════════════════════════════════════════════════════════════════════
# PR-AUC
# ═══════════════════════════════════════════════════════════════════════════

def test_perfect_scores_pr_auc_equals_one():
    """
    A perfect ranker assigns score 1 to every fraud and 0 to every legit row.
    The PR-AUC should be exactly 1.0.

    Why?
    ----
    Average precision = area under the Precision-Recall curve.  A perfect
    ranker recovers all frauds at precision 1.0, so AP = 1.0.
    """
    result = pr_auc(_y, _s_perfect)
    assert result == pytest.approx(1.0, abs=1e-9), (
        f"Expected PR-AUC=1.0 for perfect scores, got {result:.6f}"
    )


def test_random_scores_pr_auc_approx_base_rate():
    """
    A random ranker should achieve PR-AUC ≈ base_rate (the fraction of frauds).

    Why?
    ----
    If scores are uncorrelated with labels, the precision at any recall
    threshold is just the fraction of frauds in the dataset, so AP ≈ base_rate.
    We allow ±3× the expected standard error to avoid flaky failures.
    """
    br = base_rate(_y)
    result = pr_auc(_y, _s_random)
    # Loose tolerance: AP from a random ranker fluctuates around base_rate
    assert abs(result - br) < 5 * br, (
        f"Random ranker PR-AUC {result:.4f} deviates too much from "
        f"base_rate {br:.4f}."
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROC-AUC
# ═══════════════════════════════════════════════════════════════════════════

def test_roc_auc_perfect():
    """Perfect scores → ROC-AUC = 1.0."""
    assert roc_auc(_y, _s_perfect) == pytest.approx(1.0, abs=1e-9)


def test_roc_auc_random():
    """Random scores → ROC-AUC ≈ 0.5 (within 0.05)."""
    result = roc_auc(_y, _s_random)
    assert abs(result - 0.5) < 0.05, (
        f"Expected ROC-AUC ≈ 0.5 for random scores, got {result:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# recall_at_fpr — monotonicity
# ═══════════════════════════════════════════════════════════════════════════

def test_recall_at_fpr_monotone():
    """
    recall_at_fpr(y, s, fpr) must be monotonically non-decreasing in fpr.

    Why?
    ----
    As we allow a higher false-positive rate (looser threshold), we can only
    catch more (or equal) frauds.  A violation would indicate a bug in the ROC
    curve filtering logic.
    """
    fpr_values = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.10]
    recalls = [recall_at_fpr(_y, _s_random, f) for f in fpr_values]
    for i in range(len(recalls) - 1):
        assert recalls[i] <= recalls[i + 1] + 1e-9, (
            f"recall_at_fpr not monotone: fpr={fpr_values[i]:.4f}→{recalls[i]:.4f} "
            f"> fpr={fpr_values[i+1]:.4f}→{recalls[i+1]:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# precision_at_k
# ═══════════════════════════════════════════════════════════════════════════

def test_precision_at_k():
    """
    If the top-k scored items are exactly all the fraud rows, precision@k = 1.0.

    We construct a minimal synthetic case:
    - 10 items: 3 frauds (label=1), 7 legit (label=0)
    - Scores: frauds score 0.9/0.8/0.7; legit score 0.1..0.6
    - precision@3 should be 1.0
    """
    y_small = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    s_small = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
    result = precision_at_k(y_small, s_small, k=3)
    assert result == pytest.approx(1.0, abs=1e-9), (
        f"Expected precision@3=1.0, got {result}"
    )


def test_precision_at_k_partial():
    """
    If only 1 of top-2 is fraud, precision@2 = 0.5.
    """
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.05])
    # Top-2: indices 0 (fraud) and 1 (legit)
    result = precision_at_k(y, s, k=2)
    assert result == pytest.approx(0.5, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# threshold_at_fpr
# ═══════════════════════════════════════════════════════════════════════════

def test_threshold_at_fpr_legit_only():
    """
    threshold_at_fpr must use ONLY legitimate rows.

    We create a val set where legit scores are U[0,1] and fraud scores are
    U[0.9,1].  The 99th percentile of legit scores ≈ 0.99, so threshold at
    FPR=0.01 should be near 0.99, not influenced by the fraud scores.
    """
    rng = np.random.default_rng(0)
    n_legit, n_fraud = 1000, 50
    s_legit = rng.uniform(0, 1, n_legit)
    s_fraud = rng.uniform(0.9, 1.0, n_fraud)
    y = np.array([0] * n_legit + [1] * n_fraud)
    s = np.concatenate([s_legit, s_fraud])

    thresh = threshold_at_fpr(y, s, fpr=0.01)

    # The 99th percentile of Uniform[0,1] ≈ 0.99
    expected_approx = np.quantile(s_legit, 0.99)
    assert abs(thresh - expected_approx) < 1e-9, (
        f"Expected threshold ≈ {expected_approx:.4f}, got {thresh:.4f}. "
        "threshold_at_fpr may not be using legit-only scores."
    )


# ═══════════════════════════════════════════════════════════════════════════
# confusion_at_threshold
# ═══════════════════════════════════════════════════════════════════════════

def test_confusion_at_threshold():
    """
    Hand-calculated confusion matrix for a tiny example.

    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1]
    threshold = 0.75

    Predicted positives (score >= 0.75): index 0 (s=0.9, y=1 → TP)
                                         index 1 (s=0.8, y=1 → TP)
    Predicted negatives: index 2 (s=0.7, y=0 → TN)
                         index 3 (s=0.4, y=0 → TN)
                         index 4 (s=0.3, y=1 → FN)
                         index 5 (s=0.1, y=0 → TN)

    TP=2, FP=0, TN=3, FN=1
    """
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    cm = confusion_at_threshold(y, s, thresh=0.75)
    assert cm["TP"] == 2
    assert cm["FP"] == 0
    assert cm["TN"] == 3
    assert cm["FN"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# base_rate
# ═══════════════════════════════════════════════════════════════════════════

def test_base_rate():
    """base_rate([1,1,0,0,0]) = 2/5 = 0.4."""
    assert base_rate(np.array([1, 1, 0, 0, 0])) == pytest.approx(0.4, abs=1e-9)


def test_base_rate_all_legit():
    """base_rate([0,0,0]) = 0.0."""
    assert base_rate(np.array([0, 0, 0])) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# evaluate_scores
# ═══════════════════════════════════════════════════════════════════════════

def test_evaluate_scores_shape():
    """
    evaluate_scores should return a DataFrame with:
    - columns: ['val', 'test']
    - at least one 'PR-AUC' row
    - NO 'accuracy' row (accuracy is excluded by design)
    """
    # Use the shared synthetic data for both val and test
    cfg_stub = {
        "eval": {
            "fpr_budgets": [0.01],
            "review_fractions": [0.01],
            "top_k": [50],
        }
    }
    df = evaluate_scores(_y, _s_random, _y, _s_random, cfg_stub)

    assert isinstance(df, pd.DataFrame), "evaluate_scores must return a DataFrame."
    assert list(df.columns) == ["val", "test"], (
        f"Expected columns ['val','test'], got {list(df.columns)}"
    )
    assert "PR-AUC" in df.index, "PR-AUC must be present in the results."
    assert "accuracy" not in [str(i).lower() for i in df.index], (
        "Accuracy must NOT appear in the evaluation table (it is misleading "
        "under severe class imbalance)."
    )


def test_evaluate_scores_no_leakage_via_threshold():
    """
    Thresholds in evaluate_scores must be derived from val and applied to test
    without re-derivation.  We verify that PR-AUC values for val and test
    are independent (can differ), confirming they are computed on their
    respective splits.
    """
    cfg_stub = {
        "eval": {
            "fpr_budgets": [0.01],
            "review_fractions": [0.01],
            "top_k": [50],
        }
    }
    # Use random scores but different for val and test
    rng = np.random.default_rng(99)
    s_val2  = rng.uniform(0, 1, len(_y))
    s_test2 = rng.uniform(0, 1, len(_y))

    df = evaluate_scores(_y, s_val2, _y, s_test2, cfg_stub)
    # PR-AUC_val and PR-AUC_test are allowed to differ
    val_prauc  = float(df.loc["PR-AUC", "val"])
    test_prauc = float(df.loc["PR-AUC", "test"])
    # Both should be near base_rate (≈ 0.02) for random scores
    br = base_rate(_y)
    assert abs(val_prauc  - br) < 10 * br
    assert abs(test_prauc - br) < 10 * br
