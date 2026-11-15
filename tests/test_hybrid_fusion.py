"""
tests/test_hybrid_fusion.py
===========================
Unit and integration tests for Phase 2 Hybrid H2 (Score Fusion).
Author: Prachee Dhar (Task P2_04)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import get_repo_root
from src.hybrid_fusion import (
    fuse_max,
    fuse_rank_average,
    fuse_weighted,
    normalize_log_zscore,
    normalize_minmax,
    normalize_rank,
    normalize_zscore,
    sweep_fusion_weights,
)
from src.metrics import pr_auc, recall_at_fpr
from src.utils import load_scores


def test_normalize_zscore():
    rng = np.random.default_rng(42)
    ref = rng.normal(10.0, 2.0, size=1000)
    data = np.array([6.0, 10.0, 14.0])
    normed, mu, sigma = normalize_zscore(data, ref_scores=ref)
    assert np.isclose(mu, 10.0, atol=0.2)
    assert np.isclose(sigma, 2.0, atol=0.2)
    assert np.isclose(normed[1], 0.0, atol=0.1)
    assert normed[0] < normed[1] < normed[2]


def test_normalize_minmax():
    ref = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    data = np.array([10.0, 30.0, 50.0])
    normed, s_min, s_max = normalize_minmax(data, ref_scores=ref)
    assert s_min == 10.0
    assert s_max == 50.0
    assert np.allclose(normed, [0.0, 0.5, 1.0])


def test_normalize_rank():
    scores = np.array([1.5, 10.2, 3.4, 0.1])
    normed = normalize_rank(scores)
    assert len(normed) == 4
    # Expected ranking: 0.1 -> 0, 1.5 -> 1/3, 3.4 -> 2/3, 10.2 -> 1
    assert np.isclose(normed[3], 0.0)
    assert np.isclose(normed[1], 1.0)
    assert normed[3] < normed[0] < normed[2] < normed[1]


def test_fusion_operators():
    s_ae = np.array([1.0, 2.0, 3.0])
    s_gmm = np.array([4.0, 2.0, 0.0])

    # Weighted
    w_05 = fuse_weighted(s_ae, s_gmm, alpha=0.5)
    assert np.allclose(w_05, [2.5, 2.0, 1.5])

    # Max
    s_max = fuse_max(s_ae, s_gmm)
    assert np.allclose(s_max, [4.0, 2.0, 3.0])

    # Rank average
    s_rank = fuse_rank_average(s_ae, s_gmm)
    assert len(s_rank) == 3
    assert s_rank.dtype == np.float32


def test_sweep_fusion_weights_synthetic():
    rng = np.random.default_rng(123)
    n_samples = 1000
    y_val = np.zeros(n_samples, dtype=int)
    y_val[:20] = 1  # 20 frauds

    s_ae_val = rng.normal(0, 1, n_samples)
    s_ae_val[:20] += 3.0  # frauds have higher error

    s_gmm_val = rng.normal(0, 1, n_samples)
    s_gmm_val[:20] += 2.0

    y_test = y_val.copy()
    s_ae_test = s_ae_val.copy()
    s_gmm_test = s_gmm_val.copy()

    df_sweep, best_alpha = sweep_fusion_weights(
        s_ae_val, s_gmm_val, y_val, s_ae_test, s_gmm_test, y_test, norm_method="zscore"
    )
    assert len(df_sweep) == 11
    assert 0.0 <= best_alpha <= 1.0
    assert "val_pr_auc" in df_sweep.columns
    assert "test_pr_auc" in df_sweep.columns


def test_saved_hybrid_fusion_score_contracts():
    root = get_repo_root()
    val_p = root / "results" / "scores" / "hybrid_fusion_val.npy"
    test_p = root / "results" / "scores" / "hybrid_fusion_test.npy"

    assert val_p.exists(), "hybrid_fusion_val.npy missing"
    assert test_p.exists(), "hybrid_fusion_test.npy missing"

    s_val = np.load(val_p)
    s_test = np.load(test_p)

    assert s_val.shape == (56961,)
    assert s_test.shape == (56962,)
    assert s_val.dtype == np.float32
    assert s_test.dtype == np.float32
    assert not np.isnan(s_val).any()
    assert not np.isnan(s_test).any()
    assert not np.isinf(s_val).any()
    assert not np.isinf(s_test).any()
