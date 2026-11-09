"""
tests/test_dl_data.py
=====================
Unit and integration tests for Phase 2 Deep Learning Data Pipeline.
Author: Krishna Pramod Dubale (Task P2_01)
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.config import CFG
from src.data import get_features
from src.eda_dl import (
    create_dl_tensors,
    create_dl_dataloaders,
    audit_dl_leakage_and_invariants,
    analyze_feature_variances_and_dominance,
)


@pytest.fixture(scope="module")
def base_splits():
    return get_features(feature_set="base")


@pytest.fixture(scope="module")
def dl_tensors(base_splits):
    return create_dl_tensors(base_splits)


@pytest.fixture(scope="module")
def dl_loaders(dl_tensors):
    return create_dl_dataloaders(dl_tensors, batch_size=512, seed=int(CFG.get("seed", 42)))


def test_dl_tensors_pure_legitimate(base_splits, dl_tensors):
    """Assert strictly ZERO fraud in training tensors."""
    n_legit_expected = int((base_splits.y_train_all == 0).sum())
    assert len(dl_tensors["X_train_legit"]) == n_legit_expected
    assert len(dl_tensors["X_train_legit"]) == 170524


def test_dl_tensors_shapes_and_dtypes(dl_tensors):
    """Verify expected tensor dimensions and float32 dtype."""
    assert dl_tensors["X_train_legit"].shape == (170524, 31)
    assert dl_tensors["X_val_legit"].shape == (56904, 31)
    assert dl_tensors["X_val"].shape == (56961, 31)
    assert dl_tensors["y_val"].shape == (56961,)
    assert dl_tensors["X_test"].shape == (56962, 31)
    assert dl_tensors["y_test"].shape == (56962,)

    for k, t in dl_tensors.items():
        assert t.dtype == torch.float32, f"Tensor {k} has wrong dtype {t.dtype}"


def test_dl_tensors_no_nan_or_inf(dl_tensors):
    """Verify all tensors contain strictly finite numbers."""
    for k, t in dl_tensors.items():
        assert torch.isfinite(t).all(), f"Tensor {k} contains NaN or Inf values!"


def test_dl_dataloader_batch_shapes(dl_loaders):
    """Verify DataLoader batch generation and autoencoder target identity."""
    train_loader = dl_loaders["train_loader"]
    assert len(train_loader) == 334  # ceil(170524 / 512)

    batch_x, batch_target = next(iter(train_loader))
    assert batch_x.shape == (512, 31)
    assert batch_target.shape == (512, 31)
    assert torch.equal(batch_x, batch_target)


def test_dl_val_legit_early_stopping_isolation(base_splits, dl_tensors):
    """Verify validation legit tensor excludes all 57 validation fraud cases."""
    val_fraud_count = int((base_splits.y_val == 1).sum())
    assert val_fraud_count == 57
    assert len(dl_tensors["X_val_legit"]) == len(base_splits.X_val) - 57
    assert len(dl_tensors["X_val_legit"]) == 56904


def test_dl_scaler_matches_train_legit(base_splits):
    """Verify scaler mean matches training legitimate feature mean exactly."""
    train_legit_mean = np.mean(base_splits.X_train_legit, axis=0)
    # Since X_train_legit is scaled with its own mean, its mean should be ~0.0
    np.testing.assert_allclose(train_legit_mean, 0.0, atol=1e-5)


def test_dl_variance_dominance_math(base_splits):
    """Verify that training legitimate scaled variance is approximately 1.0 per coordinate."""
    train_legit_var = np.var(base_splits.X_train_legit, axis=0)
    np.testing.assert_allclose(train_legit_var, 1.0, atol=1e-3)



def test_audit_leakage_table_passes(base_splits):
    """Verify all audit table checks return 'PASS'."""
    audit_df = audit_dl_leakage_and_invariants(base_splits)
    assert (audit_df["Status"] == "PASS").all()
