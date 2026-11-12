"""
tests/test_hybrid_latent_gmm.py
==============================
Unit and integration tests for Phase 2 Hybrid H1 (``src/hybrid_latent_gmm.py``).
Author: Apoorva Choudhary (Task P2_02)

All tests use in-memory labelled synthetic stubs and NEVER write to ``results/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import get_repo_root
from src.data import Splits
from src.hybrid_latent_gmm import (
    LatentSplits,
    WaitingForInputsError,
    analyze_latent_gaussianity,
    analyze_latentgmm_failures,
    check_ae_dependencies,
    compare_with_parents,
    compute_complementarity_summary,
    compute_latent_nll,
    compute_parameter_count_comparison,
    discover_ablation_checkpoints,
    encode_matrix,
    encode_splits,
    fit_latent_gmm,
    load_vani_encoder,
    run_gmm_ablation,
    run_latent_gmm_grid_search,
    run_reg_covar_sensitivity,
    run_seed_stability_comparison,
    run_synthetic_stub_self_test,
    select_latent_gmm_config,
    summarize_gaussianity,
)
from src.metrics import pr_auc


class _TorchStubAE(nn.Module):
    """Minimal PyTorch autoencoder stub exposing ``encode()`` for offline tests."""

    def __init__(self, d_in: int = 31, latent_dim: int = 8, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _make_synthetic_splits(
    d_in: int = 31,
    n_train_legit: int = 800,
    n_train_fraud: int = 12,
    n_val_legit: int = 400,
    n_val_fraud: int = 20,
    n_test_legit: int = 400,
    n_test_fraud: int = 20,
    seed: int = 42,
) -> Splits:
    """Create labelled synthetic Splits strictly in memory."""
    rng = np.random.default_rng(seed)
    X_tr_leg = rng.normal(0.0, 1.0, size=(n_train_legit, d_in)).astype(np.float32)
    X_tr_frd = rng.normal(2.5, 1.2, size=(n_train_fraud, d_in)).astype(np.float32)
    X_tr_all = np.vstack([X_tr_leg, X_tr_frd])
    y_tr_all = np.array([0] * n_train_legit + [1] * n_train_fraud, dtype=np.int32)

    X_val = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(n_val_legit, d_in)),
            rng.normal(2.5, 1.2, size=(n_val_fraud, d_in)),
        ]
    ).astype(np.float32)
    y_val = np.array([0] * n_val_legit + [1] * n_val_fraud, dtype=np.int32)

    X_test = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(n_test_legit, d_in)),
            rng.normal(2.5, 1.2, size=(n_test_fraud, d_in)),
        ]
    ).astype(np.float32)
    y_test = np.array([0] * n_test_legit + [1] * n_test_fraud, dtype=np.int32)

    return Splits(
        X_train_legit=X_tr_leg,
        X_train_all=X_tr_all,
        y_train_all=y_tr_all,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=[f"f_{i}" for i in range(d_in)],
        scaler=None,  # type: ignore[arg-type]
    )


def test_waiting_for_inputs_when_vani_ae_absent(tmp_path: Path):
    """Fail loudly with a clear 'waiting for inputs' error when AE artifacts are absent."""
    ae_path = get_repo_root() / "src" / "autoencoder.py"
    if not ae_path.exists():
        with pytest.raises(WaitingForInputsError, match="(?i)waiting for inputs"):
            check_ae_dependencies(models_dir=tmp_path)
        with pytest.raises(WaitingForInputsError, match="(?i)waiting for inputs"):
            load_vani_encoder(models_dir=tmp_path)
    else:
        with pytest.raises(WaitingForInputsError, match="(?i)waiting for inputs"):
            check_ae_dependencies(checkpoint_path=tmp_path / "ae_missing.pt", models_dir=tmp_path)


def test_latent_splits_leakage_and_shape_guards():
    """Verify LatentSplits blocks fraud infiltration into training codes and shape mismatches."""
    rng = np.random.default_rng(7)
    y_train_all = np.array([0] * 100 + [1] * 5, dtype=np.int32)

    # Wrong row count (simulating accidental inclusion of fraud rows in Z_train_legit)
    with pytest.raises(ValueError, match="LEAKAGE GUARD"):
        LatentSplits(
            Z_train_legit=rng.normal(size=(105, 8)),
            Z_val=rng.normal(size=(50, 8)),
            y_val=np.zeros(50, dtype=np.int32),
            Z_test=rng.normal(size=(50, 8)),
            y_test=np.zeros(50, dtype=np.int32),
            latent_dim=8,
            y_train_all=y_train_all,
        )

    # NaN guard
    bad = rng.normal(size=(100, 8))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        LatentSplits(
            Z_train_legit=bad,
            Z_val=rng.normal(size=(50, 8)),
            y_val=np.zeros(50, dtype=np.int32),
            Z_test=rng.normal(size=(50, 8)),
            y_test=np.zeros(50, dtype=np.int32),
            latent_dim=8,
            y_train_all=y_train_all,
        )


def test_torch_encoder_preserves_order_and_shape():
    """Verify batched PyTorch encode() preserves exact sample order without shuffling."""
    splits = _make_synthetic_splits(d_in=31)
    enc = _TorchStubAE(d_in=31, latent_dim=8, seed=42)

    # Batched vs full-matrix forward pass must agree identically
    z_batched = encode_matrix(enc, splits.X_val, batch_size=64, device="cpu")
    with torch.no_grad():
        z_full = enc.encode(torch.from_numpy(splits.X_val)).numpy().astype(np.float64)

    assert z_batched.shape == (len(splits.X_val), 8)
    np.testing.assert_allclose(z_batched, z_full, rtol=1e-6, atol=1e-6)

    ls = encode_splits(enc, splits, batch_size=128, device="cpu", feature_set="stub", encoder_id="torch_stub")
    assert ls.latent_dim == 8
    assert len(ls.Z_train_legit) == int((splits.y_train_all == 0).sum())


def test_val_only_selection_ignores_test_labels():
    """Changing test arrays/labels must NOT alter the grid-selected configuration."""
    splits = _make_synthetic_splits(d_in=31)
    enc = _TorchStubAE(d_in=31, latent_dim=8, seed=42)
    ls1 = encode_splits(enc, splits, batch_size=256, device="cpu")
    ls2 = LatentSplits(
        Z_train_legit=ls1.Z_train_legit.copy(),
        Z_val=ls1.Z_val.copy(),
        y_val=ls1.y_val.copy(),
        Z_test=np.zeros_like(ls1.Z_test),
        y_test=np.zeros_like(ls1.y_test),
        latent_dim=ls1.latent_dim,
        y_train_all=ls1.y_train_all,
    )

    g1 = run_latent_gmm_grid_search(ls1, cov_types=("full", "diag"), k_values=(1, 2), reg_covars=(1e-4,), subsample_size=None, seed=42)
    g2 = run_latent_gmm_grid_search(ls2, cov_types=("full", "diag"), k_values=(1, 2), reg_covars=(1e-4,), subsample_size=None, seed=42)
    cfg1, _ = select_latent_gmm_config(g1, selection_criterion="val_pr_auc")
    cfg2, _ = select_latent_gmm_config(g2, selection_criterion="val_pr_auc")

    assert cfg1 == cfg2


def test_latent_gmm_scores_direction_and_contract():
    """Latent NLL scores must be 1-D float32, finite, and higher for anomalies."""
    splits = _make_synthetic_splits(d_in=31)
    enc = _TorchStubAE(d_in=31, latent_dim=8, seed=42)
    ls = encode_splits(enc, splits, batch_size=256, device="cpu")

    gmm = fit_latent_gmm(ls.Z_train_legit, n_components=2, covariance_type="full", reg_covar=1e-4, random_state=42)
    s_val = compute_latent_nll(gmm, ls.Z_val)
    s_test = compute_latent_nll(gmm, ls.Z_test)

    assert s_val.dtype == np.float32 and s_val.shape == (len(ls.y_val),)
    assert s_test.dtype == np.float32 and s_test.shape == (len(ls.y_test),)
    assert np.isfinite(s_val).all() and np.isfinite(s_test).all()

    assert float(s_test[ls.y_test == 1].mean()) > float(s_test[ls.y_test == 0].mean())
    assert pr_auc(ls.y_test, s_test) > float(ls.y_test.mean())


def test_task_b_and_c_analytics_on_stubs():
    """Verify parameter count arithmetic, evidence studies, parent comparison, and ablation on stubs."""
    splits = _make_synthetic_splits(d_in=31)
    enc8 = _TorchStubAE(d_in=31, latent_dim=8, seed=42)
    enc4 = _TorchStubAE(d_in=31, latent_dim=4, seed=43)
    ls8 = encode_splits(enc8, splits, batch_size=256, device="cpu")
    ls4 = encode_splits(enc4, splits, batch_size=256, device="cpu")

    # (i) Parameter arithmetic check
    param_df = compute_parameter_count_comparison(ambient_dims=(30, 31), latent_dims=(4, 8), k_values=(12,))
    full_d30 = int(param_df.query("dimension == 30 and covariance_type == 'full' and n_components == 12")["free_parameters"].iloc[0])
    full_d31 = int(param_df.query("dimension == 31 and covariance_type == 'full' and n_components == 12")["free_parameters"].iloc[0])
    full_d8 = int(param_df.query("dimension == 8 and covariance_type == 'full' and n_components == 12")["free_parameters"].iloc[0])
    assert full_d30 == 5951
    assert full_d31 == 6335
    assert full_d8 == 539

    # (ii) reg_covar sensitivity
    reg_df = run_reg_covar_sensitivity(
        splits.X_train_legit, splits.X_val, splits.y_val, ls8.Z_train_legit, ls8.Z_val,
        n_components=2, reg_values=(1e-5, 1e-2), cov_types=("full", "diag"), subsample_size=500, seed=42,
    )
    assert len(reg_df) == 8
    assert (reg_df["mean_condition_number"].dropna() >= 1.0).all()

    # (iii) Seed stability
    seed_df = run_seed_stability_comparison(
        splits.X_train_legit, splits.X_val, splits.y_val, ls8.Z_train_legit, ls8.Z_val,
        ambient_cfg={"covariance_type": "diag", "n_components": 2, "reg_covar": 1e-2},
        latent_cfg={"covariance_type": "full", "n_components": 2, "reg_covar": 1e-4},
        seeds=(42, 43, 44),
    )
    assert len(seed_df) == 6

    # (iv) Gaussianity stats
    gauss_df = analyze_latent_gaussianity(splits.X_train_legit, ls8.Z_train_legit, feature_names=splits.feature_names)
    assert len(gauss_df) == 31 + 8
    assert {"skewness", "excess_kurtosis", "shapiro_w"}.issubset(gauss_df.columns)

    # (v) Failure & parent comparison
    gmm = fit_latent_gmm(ls8.Z_train_legit, n_components=2, covariance_type="full", reg_covar=1e-4, random_state=42)
    s_val = compute_latent_nll(gmm, ls8.Z_val)
    s_test = compute_latent_nll(gmm, ls8.Z_test)
    stub_parent = {"ae_stub": {"val": s_val * 0.9, "test": s_test * 0.9}}
    fail_df = analyze_latentgmm_failures(ls8.y_val, s_val, parent_scores={"ae_stub": stub_parent["ae_stub"]["val"]}, Z_split=ls8.Z_val, n_missed=5)
    assert len(fail_df) == 5

    ci_df, paired_df = compare_with_parents(ls8.y_val, s_val, ls8.y_test, s_test, stub_parent, n_boot=100, seed=42)
    assert len(ci_df) == 2 and len(paired_df) == 1

    # Task C ablation
    runs_df, summary_df = run_gmm_ablation(
        {4: ls4, 8: ls8},
        cov_types=("full", "diag"),
        k_values=(1, 2),
        seeds=(42, 43, 44),
        subsample_size=None,
    )
    assert len(runs_df) == 2 * 2 * 2 * 3
    assert len(summary_df) == 2 * 2 * 2


def test_synthetic_stub_self_test_writes_nothing_to_results():
    """Confirm self-test passes end-to-end and never writes artifacts to results/."""
    results_dir = get_repo_root() / "results"
    before = sorted(p.relative_to(results_dir) for p in results_dir.rglob("*") if p.is_file())
    out = run_synthetic_stub_self_test(seed=42)
    after = sorted(p.relative_to(results_dir) for p in results_dir.rglob("*") if p.is_file())

    assert before == after
    assert out["test_pr_auc"] > 0.5


def test_gaussianity_summary_and_complementarity_helpers():
    """Verify summary helpers for Gaussianity and alert complementarity."""
    splits = _make_synthetic_splits(d_in=31)
    enc8 = _TorchStubAE(d_in=31, latent_dim=8, seed=42)
    ls8 = encode_splits(enc8, splits, batch_size=256, device="cpu")

    gauss_df = analyze_latent_gaussianity(splits.X_train_legit, ls8.Z_train_legit, feature_names=splits.feature_names)
    summary_df = summarize_gaussianity(gauss_df)
    assert len(summary_df) == 2
    assert set(summary_df["space"]) == {"ambient_d31", "latent_d8"}

    gmm = fit_latent_gmm(ls8.Z_train_legit, n_components=2, covariance_type="full", reg_covar=1e-4, random_state=42)
    s_val = compute_latent_nll(gmm, ls8.Z_val)
    comp_df = compute_complementarity_summary(ls8.y_val, s_val, {"ae_stub": s_val * 0.9})
    assert len(comp_df) == 3
    assert (comp_df["h1_caught"] + comp_df["parent_only_caught"] + comp_df["missed_by_both"] == 20).all()


def test_vani_checkpoints_and_saved_hybrid_scores():
    """Verify Vani's AE checkpoints load cleanly and saved H1 scores meet the contract."""
    root = get_repo_root()
    ae_ckpt = root / "results/models/ae_best.pt"
    if not ae_ckpt.exists():
        pytest.skip("Vani's ae_best.pt not present.")

    enc, meta = load_vani_encoder(ae_ckpt)
    assert hasattr(enc, "encode")
    assert enc.latent_dim == 12
    assert "ae_best.pt" in meta["checkpoint_path"]

    ablation_ckpts = discover_ablation_checkpoints()
    assert set([2, 4, 8, 16]).issubset(ablation_ckpts.keys())

    val_path = root / "results/scores/hybrid_latentgmm_val.npy"
    test_path = root / "results/scores/hybrid_latentgmm_test.npy"
    if not (val_path.exists() and test_path.exists()):
        pytest.skip("Day 2 H1 score files not yet generated.")

    import pandas as pd

    y_val = pd.read_csv(root / "data/processed/val.csv", index_col=0)["Class"].values
    y_test = pd.read_csv(root / "data/processed/test.csv", index_col=0)["Class"].values
    s_val = np.load(val_path)
    s_test = np.load(test_path)

    assert s_val.dtype == np.float32 and s_val.shape == (len(y_val),)
    assert s_test.dtype == np.float32 and s_test.shape == (len(y_test),)
    assert np.isfinite(s_val).all() and np.isfinite(s_test).all()
    assert pr_auc(y_test, s_test) > float(y_test.mean())
    assert float(s_test[y_test == 1].mean()) > float(s_test[y_test == 0].mean())

