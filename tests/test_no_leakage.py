"""
tests/test_no_leakage.py
========================
Checks that the chronological split and feature extraction pipeline do NOT
leak information from the future into the past or from fraud rows into the
training-legit feature matrix.

Tests
-----
1. ``test_time_ordering``
   train_max_time < val_min_time < test_min_time.
2. ``test_scaler_mean_equals_train_legit_mean``
   The scaler was fit on training-legit rows only, so its ``mean_`` should
   equal the unscaled column means of ``X_train_legit`` before scaling
   (i.e. the mean of the train-legit raw features).
3. ``test_no_fraud_in_X_train_legit``
   ``X_train_legit`` must contain ZERO fraud rows.

All tests share a module-level ``Splits`` fixture so the CSV is loaded once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CFG, get_repo_root
from src.data import get_features, load_raw, chronological_split

_ROOT = get_repo_root()
_PROCESSED = _ROOT / CFG["paths"]["processed"]


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixture (module-scoped = run once)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def splits_fixture():
    """Load data and return (splits, train_df, val_df, test_df)."""
    splits = get_features("base")
    df = load_raw()
    train_df, val_df, test_df = chronological_split(df)
    return splits, train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — Time ordering
# ═══════════════════════════════════════════════════════════════════════════

def test_time_ordering(splits_fixture):
    """
    The chronological split must satisfy:
        max(train.Time) < min(val.Time) < min(test.Time)

    This guarantees no future data leaks into training.

    Why?
    ----
    If we accidentally shuffle before splitting, a transaction from hour 47
    could end up in the training set while a transaction from hour 1 is in
    the test set — the model would then 'see the future'.
    """
    _, train_df, val_df, test_df = splits_fixture

    train_max = float(train_df["Time"].max())
    val_min   = float(val_df["Time"].min())
    val_max   = float(val_df["Time"].max())
    test_min  = float(test_df["Time"].min())

    assert train_max <= val_min, (
        f"Time ordering violated: train_max ({train_max:.1f}) > val_min ({val_min:.1f}). "
        "The split must be strictly chronological (no shuffling)."
    )
    assert val_max <= test_min, (
        f"Time ordering violated: val_max ({val_max:.1f}) > test_min ({test_min:.1f}). "
        "Validation data must precede test data in time."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — Scaler mean equals train-legit mean
# ═══════════════════════════════════════════════════════════════════════════

def test_scaler_mean_equals_train_legit_mean(splits_fixture):
    """
    ``scaler.mean_`` should equal the column-wise mean of the UNscaled
    training-legit feature matrix.

    Why?
    ----
    StandardScaler.fit(X) stores the mean of X in scaler.mean_.  If the
    scaler was accidentally fit on *all* training rows (including fraud), its
    mean would differ from the legit-only mean.  This test catches that.

    Approach:
    We reconstruct the unscaled train-legit matrix by reversing the scaling
    (X_train_legit_unscaled = X_train_legit * scaler.scale_ + scaler.mean_)
    and check that its column mean equals scaler.mean_.
    """
    splits, _, _, _ = splits_fixture
    scaler = splits.scaler
    X_legit = splits.X_train_legit.astype(np.float64)

    # Reverse the scaling: x_raw = x_scaled * scale + mean
    X_unscaled = X_legit * scaler.scale_ + scaler.mean_

    computed_mean = X_unscaled.mean(axis=0)

    np.testing.assert_allclose(
        computed_mean,
        scaler.mean_,
        rtol=1e-4,
        atol=1e-4,
        err_msg=(
            "scaler.mean_ does not match the column means of X_train_legit. "
            "This indicates the scaler was not fit exclusively on training-legit rows."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — No fraud in X_train_legit
# ═══════════════════════════════════════════════════════════════════════════

def test_no_fraud_in_X_train_legit(splits_fixture):
    """
    ``X_train_legit`` must have the same row count as the number of Class==0
    rows in the training DataFrame — i.e. it contains ZERO fraud rows.

    Why?
    ----
    Anomaly models are trained unsupervised on legitimate transactions only.
    Including even one fraud row during training would contaminate the
    learned density/boundary, violating the semi-supervised anomaly detection
    paradigm the project requires.
    """
    splits, train_df, _, _ = splits_fixture
    n_legit_train = int((train_df["Class"] == 0).sum())

    assert splits.X_train_legit.shape[0] == n_legit_train, (
        f"X_train_legit has {splits.X_train_legit.shape[0]} rows but "
        f"the training split has {n_legit_train} legitimate rows. "
        "There may be fraud rows included in X_train_legit."
    )
