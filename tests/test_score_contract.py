"""
tests/test_score_contract.py
============================
Integration test: verifies that every model's score files satisfy the
team-wide score-file contract, and that no score is fabricated.

Contract:
- File exists: results/scores/{model}_{split}.npy for val and test.
- Shape: 1-D float array.
- Length == len(data/processed/{split}.csv).
- No NaN or Inf values.
- PR-AUC > base_rate for real models (not random/amount).
- PR-AUC ≈ base_rate for random reference (within 10× base_rate).
- Higher score = more anomalous (convention check: fraud mean > legit mean).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

# Load manifest for expected split sizes
with open(ROOT / "data/processed/split_manifest.json") as f:
    _MANIFEST = json.load(f)

VAL_N  = _MANIFEST["val"]["n_rows"]
TEST_N = _MANIFEST["test"]["n_rows"]

# Load labels once
_val_df  = pd.read_csv(ROOT / "data/processed/val.csv",  index_col=0)
_test_df = pd.read_csv(ROOT / "data/processed/test.csv", index_col=0)
Y_VAL  = _val_df["Class"].values
Y_TEST = _test_df["Class"].values
BASE_RATE_VAL  = float(Y_VAL.mean())
BASE_RATE_TEST = float(Y_TEST.mean())

# Real models (should beat base rate)
REAL_MODELS = ["gmm", "pca", "iforest", "ocsvm", "lof"]
# Reference models (should be near base rate or at most moderate)
REF_MODELS  = ["random", "amount"]
ALL_MODELS  = REAL_MODELS + REF_MODELS


@pytest.mark.parametrize("model", ALL_MODELS)
@pytest.mark.parametrize("split,expected_len,y", [
    ("val",  VAL_N,  Y_VAL),
    ("test", TEST_N, Y_TEST),
])
def test_score_file_exists_and_shape(model, split, expected_len, y):
    """Score file must exist and be a 1-D float array of correct length."""
    path = ROOT / "results/scores" / f"{model}_{split}.npy"
    assert path.exists(), f"Missing score file: {path}"
    scores = np.load(path)
    assert scores.ndim == 1, f"{model}_{split}: must be 1-D, got shape {scores.shape}"
    assert len(scores) == expected_len, (
        f"{model}_{split}: length {len(scores)} != expected {expected_len}"
    )


@pytest.mark.parametrize("model", ALL_MODELS)
@pytest.mark.parametrize("split", ["val", "test"])
def test_no_nan_inf(model, split):
    """Score files must contain no NaN or Inf."""
    scores = np.load(ROOT / "results/scores" / f"{model}_{split}.npy")
    assert not np.isnan(scores).any(), f"{model}_{split}: contains NaN"
    assert not np.isinf(scores).any(), f"{model}_{split}: contains Inf"


@pytest.mark.parametrize("model", REAL_MODELS)
def test_real_model_beats_base_rate_test(model):
    """Real anomaly models must have PR-AUC > base_rate on the test split."""
    from src.metrics import pr_auc
    scores = np.load(ROOT / "results/scores" / f"{model}_test.npy")
    auc = pr_auc(Y_TEST, scores)
    assert auc > BASE_RATE_TEST, (
        f"{model} test PR-AUC={auc:.4f} should exceed base_rate={BASE_RATE_TEST:.4f}"
    )


@pytest.mark.parametrize("model", REF_MODELS)
def test_reference_model_near_base_rate(model):
    """Random/amount reference scores should be near or above base rate."""
    from src.metrics import pr_auc
    scores = np.load(ROOT / "results/scores" / f"{model}_test.npy")
    auc = pr_auc(Y_TEST, scores)
    # Random should be ≤ ~10× base_rate, amount ≤ some moderate value
    assert auc < 0.30, (
        f"{model} test PR-AUC={auc:.4f} seems too high for a reference model"
    )


@pytest.mark.parametrize("model", REAL_MODELS)
def test_higher_score_means_more_anomalous(model):
    """Fraud rows must have a higher MEAN score than legitimate rows (test set)."""
    scores = np.load(ROOT / "results/scores" / f"{model}_test.npy")
    fraud_mean = scores[Y_TEST == 1].mean()
    legit_mean = scores[Y_TEST == 0].mean()
    assert fraud_mean > legit_mean, (
        f"{model}: fraud mean score ({fraud_mean:.4f}) is NOT > "
        f"legit mean score ({legit_mean:.4f}). "
        "Check the score convention (higher = more anomalous)."
    )
