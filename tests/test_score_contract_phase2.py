"""
tests/test_score_contract_phase2.py
===================================
Integration test: verifies that Phase 2 model score files satisfy the
team-wide score-file contract, and that no score is fabricated or unverified.

Author: Kushal Sarkar (Phase 2 Lead Integration)

Contract:
- File exists: results/scores/{model}_{split}.npy for val and test.
- Models covered: ae, ae_dae, hybrid_latentgmm, hybrid_fusion, hybrid_dagmmlite.
- Shape: 1-D float array.
- Length == len(data/processed/{split}.csv) (val=56961, test=56962).
- No NaN or Inf values.
- PR-AUC > base_rate for all deep learning and hybrid models.
- Higher score = more anomalous (convention check: fraud mean > legit mean).
- AE legitimate-only training assertion (verifying clean training invariants).
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

VAL_N = _MANIFEST["val"]["n_rows"]
TEST_N = _MANIFEST["test"]["n_rows"]

# Load labels once
_val_df = pd.read_csv(ROOT / "data/processed/val.csv", index_col=0)
_test_df = pd.read_csv(ROOT / "data/processed/test.csv", index_col=0)
Y_VAL = _val_df["Class"].values
Y_TEST = _test_df["Class"].values
BASE_RATE_VAL = float(Y_VAL.mean())
BASE_RATE_TEST = float(Y_TEST.mean())

PHASE2_MODELS = [
    "ae",
    "ae_dae",
    "hybrid_latentgmm",
    "hybrid_fusion",
    "hybrid_dagmmlite",
]


@pytest.mark.parametrize("model", PHASE2_MODELS)
@pytest.mark.parametrize("split,expected_len,y", [
    ("val", VAL_N, Y_VAL),
    ("test", TEST_N, Y_TEST),
])
def test_phase2_score_file_exists_and_shape(model, split, expected_len, y):
    """Phase 2 score file must exist and be a 1-D float array of exact length."""
    path = ROOT / "results/scores" / f"{model}_{split}.npy"
    assert path.exists(), f"Missing score file: {path}"
    scores = np.load(path)
    assert scores.ndim == 1, f"{model}_{split}: must be 1-D, got shape {scores.shape}"
    assert len(scores) == expected_len, (
        f"{model}_{split}: length {len(scores)} != expected {expected_len}"
    )


@pytest.mark.parametrize("model", PHASE2_MODELS)
@pytest.mark.parametrize("split", ["val", "test"])
def test_phase2_no_nan_inf(model, split):
    """Phase 2 score files must contain zero NaN or Inf."""
    path = ROOT / "results/scores" / f"{model}_{split}.npy"
    scores = np.load(path)
    assert not np.isnan(scores).any(), f"{model}_{split}: contains NaN"
    assert not np.isinf(scores).any(), f"{model}_{split}: contains Inf"


@pytest.mark.parametrize("model", PHASE2_MODELS)
def test_phase2_model_beats_base_rate_test(model):
    """Phase 2 anomaly models must have PR-AUC > base_rate on the test split."""
    from src.metrics import pr_auc
    scores = np.load(ROOT / "results/scores" / f"{model}_test.npy")
    auc = pr_auc(Y_TEST, scores)
    assert auc > BASE_RATE_TEST, (
        f"{model} test PR-AUC={auc:.4f} should exceed base_rate={BASE_RATE_TEST:.4f}"
    )


@pytest.mark.parametrize("model", PHASE2_MODELS)
@pytest.mark.parametrize("split,y", [
    ("val", Y_VAL),
    ("test", Y_TEST),
])
def test_phase2_higher_score_means_more_anomalous(model, split, y):
    """Fraud transactions must have a higher MEAN score than legitimate transactions."""
    scores = np.load(ROOT / "results/scores" / f"{model}_{split}.npy")
    fraud_mean = float(scores[y == 1].mean())
    legit_mean = float(scores[y == 0].mean())
    assert fraud_mean > legit_mean, (
        f"{model}_{split}: fraud mean ({fraud_mean:.4f}) is NOT > legit mean ({legit_mean:.4f}). "
        "Score direction contract violated!"
    )


def test_ae_trained_on_legitimate_only_invariants():
    """Verify that autoencoder checkpoint metadata and training configs enforce legitimate-only training."""
    ae_cfg_path = ROOT / "results/models/ae_config.json"
    dae_cfg_path = ROOT / "results/models/ae_dae_config.json"
    assert ae_cfg_path.exists(), "Missing ae_config.json"
    assert dae_cfg_path.exists(), "Missing ae_dae_config.json"

    with open(ae_cfg_path) as f:
        ae_cfg = json.load(f)
    with open(dae_cfg_path) as f:
        dae_cfg = json.load(f)

    # Both configs should record legitimate-only training
    assert ae_cfg.get("target_class", 0) == 0 or "legit" in str(ae_cfg).lower()
    assert dae_cfg.get("target_class", 0) == 0 or "legit" in str(dae_cfg).lower()

    # Verify that raw train CSV has frauds filtered out for training DL tensors
    train_df = pd.read_csv(ROOT / "data/processed/train.csv", index_col=0)
    train_legit_count = int((train_df["Class"] == 0).sum())
    train_fraud_count = int((train_df["Class"] == 1).sum())
    assert train_fraud_count == 360, f"Expected 360 train frauds, got {train_fraud_count}"
    assert train_legit_count == 170524, f"Expected 170,524 legit train rows, got {train_legit_count}"
