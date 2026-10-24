"""
tests/test_evaluation.py
========================
Unit tests for ``src/evaluation.py`` — Krishna's evaluation toolkit.

All tests use synthetic stub data (never real results) to verify the
correctness of bootstrap CI, threshold policies, cost curves, and the
comparison pipeline.  Stubs follow the score-file contract:
    - 1-D float32 arrays
    - Higher score == more anomalous
    - Labels are 0 (legit) / 1 (fraud)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import pr_auc, roc_auc, recall_at_fpr, precision_at_k
from src.evaluation import (
    bootstrap_ci,
    paired_bootstrap_diff,
    threshold_policies_fpr,
    threshold_policies_capacity,
    threshold_max_f1_val,
    cost_curve,
    temporal_stability,
    compare_models,
    calibrate_scores_platt,
    calibrate_scores_isotonic,
)

# ── Synthetic stub data (NEVER saved to results/) ────────────────────────
RNG = np.random.default_rng(42)
N = 5_000
N_FRAUD = 50  # ~1% base rate
N_LEGIT = N - N_FRAUD

# Labels
_y = np.array([1] * N_FRAUD + [0] * N_LEGIT)

# Scores: model A is decent (frauds scored higher), model B is random
_s_a = np.concatenate([
    RNG.normal(3.0, 1.0, N_FRAUD),   # fraud scores ~ N(3,1)
    RNG.normal(0.0, 1.0, N_LEGIT),   # legit scores ~ N(0,1)
])
_s_b = RNG.uniform(0, 1, N)  # random model

# Perfect scores
_s_perfect = _y.astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapCI:
    """Tests for bootstrap_ci()."""

    def test_perfect_scores_ci_contains_one(self):
        """Perfect scores → PR-AUC = 1.0 and CI should include 1.0."""
        result = bootstrap_ci(pr_auc, _y, _s_perfect, n=200, seed=42)
        assert result["point"] == pytest.approx(1.0, abs=1e-6)
        assert result["ci_hi"] == pytest.approx(1.0, abs=1e-6)
        # CI should be very tight for perfect scores
        assert result["ci_lo"] > 0.95

    def test_ci_contains_point_estimate(self):
        """The CI should bracket the point estimate."""
        result = bootstrap_ci(pr_auc, _y, _s_a, n=500, seed=42)
        assert result["ci_lo"] <= result["point"] <= result["ci_hi"]

    def test_ci_width_positive(self):
        """CI width should be > 0 for noisy scores."""
        result = bootstrap_ci(pr_auc, _y, _s_a, n=500, seed=42)
        assert result["ci_hi"] - result["ci_lo"] > 0

    def test_roc_auc_ci(self):
        """Can compute CI for ROC-AUC."""
        result = bootstrap_ci(roc_auc, _y, _s_a, n=200, seed=42)
        assert 0.5 < result["point"] < 1.0
        assert result["ci_lo"] < result["ci_hi"]

    def test_recall_at_fpr_ci(self):
        """Can compute CI with extra kwargs (fpr=0.01)."""
        result = bootstrap_ci(
            recall_at_fpr, _y, _s_a, n=200, seed=42, fpr=0.01
        )
        assert 0.0 <= result["point"] <= 1.0
        assert result["ci_lo"] <= result["ci_hi"]

    def test_boot_values_length(self):
        """boot_values should have n entries."""
        n = 300
        result = bootstrap_ci(pr_auc, _y, _s_a, n=n, seed=42)
        assert len(result["boot_values"]) == n

    def test_reproducibility(self):
        """Same seed → same CI."""
        r1 = bootstrap_ci(pr_auc, _y, _s_a, n=100, seed=99)
        r2 = bootstrap_ci(pr_auc, _y, _s_a, n=100, seed=99)
        assert r1["ci_lo"] == r2["ci_lo"]
        assert r1["ci_hi"] == r2["ci_hi"]


# ═══════════════════════════════════════════════════════════════════════════
# Paired Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

class TestPairedBootstrap:
    """Tests for paired_bootstrap_diff()."""

    def test_same_model_not_significant(self):
        """Comparing a model to itself → diff = 0, not significant."""
        result = paired_bootstrap_diff(_y, _s_a, _s_a, pr_auc, n=200, seed=42)
        assert result["diff_point"] == pytest.approx(0.0, abs=1e-9)
        # CI should include 0 (not significant)
        assert not result["significant"]

    def test_good_vs_random_significant(self):
        """A good model vs random should show a significant difference."""
        result = paired_bootstrap_diff(_y, _s_a, _s_b, pr_auc, n=500, seed=42)
        # Model A should be better → positive diff
        assert result["diff_point"] > 0
        assert result["significant"]

    def test_ci_brackets_diff(self):
        """CI should bracket the point estimate of the diff."""
        result = paired_bootstrap_diff(_y, _s_a, _s_b, pr_auc, n=300, seed=42)
        assert result["ci_lo"] <= result["diff_point"] <= result["ci_hi"]


# ═══════════════════════════════════════════════════════════════════════════
# Threshold Policies
# ═══════════════════════════════════════════════════════════════════════════

class TestThresholdPolicies:
    """Tests for threshold policy functions."""

    def test_fpr_policy_shape(self):
        """FPR policy table has correct number of rows."""
        budgets = [0.001, 0.005, 0.01]
        df = threshold_policies_fpr(_y, _s_a, _y, _s_a, budgets)
        assert len(df) == len(budgets)
        for col in ["threshold", "TP", "FP", "FN", "precision", "recall", "F1"]:
            assert col in df.columns

    def test_fpr_policy_precision_recall_valid(self):
        """Precision and recall should be in [0, 1]."""
        df = threshold_policies_fpr(_y, _s_a, _y, _s_a, [0.01])
        assert 0 <= df.iloc[0]["precision"] <= 1
        assert 0 <= df.iloc[0]["recall"] <= 1

    def test_capacity_policy_shape(self):
        """Capacity policy table has correct number of rows."""
        fracs = [0.001, 0.005]
        ks = [50, 100]
        df = threshold_policies_capacity(_y, _s_a, _y, _s_a, fracs, ks)
        assert len(df) == len(fracs) + len(ks)

    def test_max_f1_returns_valid_threshold(self):
        """max-F1 threshold should be within score range."""
        result = threshold_max_f1_val(_y, _s_a, _y, _s_a)
        assert _s_a.min() <= result["threshold"] <= _s_a.max()
        assert 0 <= result["val_f1"] <= 1
        assert 0 <= result["eval_f1"] <= 1

    def test_threshold_frozen_from_val(self):
        """
        Thresholds derived from val must NOT change when test data changes.

        We call the function twice with different test data but same val data;
        the thresholds should be identical.
        """
        rng = np.random.default_rng(99)
        s_test_alt = rng.uniform(-2, 5, N)

        df1 = threshold_policies_fpr(_y, _s_a, _y, _s_a, [0.01])
        df2 = threshold_policies_fpr(_y, _s_a, _y, s_test_alt, [0.01])

        # Thresholds should be identical (derived from val only)
        assert df1.iloc[0]["threshold"] == pytest.approx(
            df2.iloc[0]["threshold"], abs=1e-12
        )

    def test_tp_fp_fn_sum_consistent(self):
        """TP + FN should equal total fraud count; TP + FP = n_alerts."""
        df = threshold_policies_fpr(_y, _s_a, _y, _s_a, [0.01])
        row = df.iloc[0]
        assert row["TP"] + row["FN"] == _y.sum()
        assert row["TP"] + row["FP"] == row["n_alerts"]


# ═══════════════════════════════════════════════════════════════════════════
# Cost Curve
# ═══════════════════════════════════════════════════════════════════════════

class TestCostCurve:
    """Tests for cost_curve()."""

    def test_cost_curve_shape(self):
        """Output has one row per cost ratio."""
        ratios = [10, 50, 100]
        df = cost_curve(_y, _s_a, _y, _s_a, ratios)
        assert len(df) == len(ratios)
        assert "cost_ratio" in df.columns
        assert "optimal_threshold" in df.columns

    def test_higher_ratio_lower_threshold(self):
        """
        As cost ratio increases (missed fraud gets more expensive),
        the optimal threshold should decrease (flag more aggressively).
        """
        ratios = [10, 100, 500]
        df = cost_curve(_y, _s_a, _y, _s_a, ratios, n_thresholds=200)
        thresholds = df["optimal_threshold"].values
        # With higher cost ratio, threshold should generally be lower or equal
        # (We allow slight non-monotonicity due to discrete threshold sweep)
        assert thresholds[0] >= thresholds[-1] - 0.5  # generous tolerance


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Stability
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalStability:
    """Tests for temporal_stability()."""

    def test_returns_n_blocks(self):
        """Output has exactly n_blocks rows."""
        time_col = np.arange(N, dtype=float)
        df = temporal_stability(_y, _s_a, time_col, n_blocks=5)
        assert len(df) == 5

    def test_nan_for_few_frauds(self):
        """
        Blocks with too few frauds should have NaN metric values.
        """
        # All frauds in first 50 rows, rest legit → later blocks have 0 fraud
        y_skewed = np.zeros(N, dtype=int)
        y_skewed[:20] = 1
        time_col = np.arange(N, dtype=float)
        s_skewed = RNG.uniform(0, 1, N)

        df = temporal_stability(
            y_skewed, s_skewed, time_col, n_blocks=5, min_fraud_per_block=5
        )
        # Later blocks should have NaN metric
        later_blocks = df[df["n_fraud"] < 5]
        assert later_blocks["metric"].isna().all()


# ═══════════════════════════════════════════════════════════════════════════
# Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

class TestCompareModels:
    """Tests for compare_models()."""

    def test_compare_returns_dataframe(self):
        """compare_models returns a DataFrame indexed by model name."""
        model_scores = {
            "model_a": {"val": _s_a, "test": _s_a},
            "model_b": {"val": _s_b, "test": _s_b},
        }
        df = compare_models(
            model_scores, _y, _y,
            fpr_budgets=[0.01], review_fractions=[0.01],
            n_boot=100, seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert "model_a" in df.index
        assert "model_b" in df.index
        assert "PR-AUC_test" in df.columns
        assert "vs_best" in df.columns

    def test_best_model_ranked_first(self):
        """Best model by PR-AUC should be first."""
        model_scores = {
            "good": {"val": _s_a, "test": _s_a},
            "random": {"val": _s_b, "test": _s_b},
        }
        df = compare_models(
            model_scores, _y, _y,
            fpr_budgets=[0.01], review_fractions=[0.01],
            n_boot=100, seed=42,
        )
        assert df.index[0] == "good"
        assert df.loc["good", "vs_best"] == "best"


# ═══════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibration:
    """Tests for calibration functions."""

    def test_platt_returns_probabilities(self):
        """Platt scaling should produce values in [0, 1]."""
        probs, model = calibrate_scores_platt(_s_a, _y, _s_a)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert len(probs) == len(_s_a)

    def test_isotonic_returns_probabilities(self):
        """Isotonic regression should produce values in [0, 1]."""
        probs, model = calibrate_scores_isotonic(_s_a, _y, _s_a)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert len(probs) == len(_s_a)
