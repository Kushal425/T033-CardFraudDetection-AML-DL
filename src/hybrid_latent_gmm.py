"""
src/hybrid_latent_gmm.py
========================
Phase 2 Hybrid H1: Latent-Space Gaussian Mixture Model (GMM) Density Scoring,
Theoretical Evidence Benchmarks, and GMM Ablations for Team T033.

Author: Apoorva Choudhary (Task P2_02)

Theoretical & Operational Contract
----------------------------------
1. Depends on Vani's autoencoder module (``src/autoencoder.py`` exposing ``encode()``)
   and trained weights under ``results/models/ae_*.pt``. If absent, fail loudly
   with :class:`WaitingForInputsError` ("waiting for inputs").
2. Latent GMM is fitted **exclusively** on legitimate training latent vectors
   ``Z_train_legit`` (``Class == 0``). Fraud labels never enter EM fitting.
3. Component count ``K``, ``covariance_type``, and ``reg_covar`` are chosen
   using ONLY train-legit BIC and validation metrics (never test labels).
4. Anomaly score convention: ``s(z) = -ln p(z)`` (1-D ``float32`` array;
   higher score == lower latent density == more anomalous).
5. Output score files for Prachee's downstream hybrids (H2/H3) and Kushal's
   final evaluation:
   - ``results/scores/hybrid_latentgmm_val.npy``
   - ``results/scores/hybrid_latentgmm_test.npy``
6. Synthetic-stub self-tests run strictly in memory and NEVER write to ``results/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Ensure repository root is importable when script is executed directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from src.config import CFG, get_repo_root
from src.data import Splits, get_features
from src.evaluation import bootstrap_ci, paired_bootstrap_diff
from src.gmm_model import calculate_free_params, compute_anomaly_scores, fit_gmm
from src.metrics import evaluate_scores, pr_auc, recall_at_fpr, roc_auc
from src.utils import get_logger, load_scores, save_fig, save_scores, save_table, set_seed

_logger = get_logger(__name__)
_ROOT: Path = get_repo_root()


class WaitingForInputsError(RuntimeError):
    """Raised when required upstream Phase 2 Autoencoder artifacts are missing."""


@dataclasses.dataclass
class LatentSplits:
    """
    Aligned latent-space arrays and ground-truth labels for Phase 2 H1.

    Attributes
    ----------
    Z_train_legit : np.ndarray, shape (n_train_legit, d_latent)
        Encoded legitimate training transactions ONLY (Class == 0).
    Z_val : np.ndarray, shape (n_val, d_latent)
        Encoded validation split preserving exact chronological row order.
    y_val : np.ndarray, shape (n_val,)
        Validation binary labels (used only for model/threshold selection).
    Z_test : np.ndarray, shape (n_test, d_latent)
        Encoded test split preserving exact chronological row order.
    y_test : np.ndarray, shape (n_test,)
        Test binary labels (touched only once for final evaluation).
    latent_dim : int
        Dimensionality of the encoder bottleneck.
    feature_set : str
        Name of the upstream tabular feature set (e.g. 'base', 'engineered').
    encoder_id : str
        Identifier of the encoder model/checkpoint used to produce codes.
    y_train_all : np.ndarray, optional
        If provided, verified so ``len(Z_train_legit) == (y_train_all == 0).sum()``.
    """

    Z_train_legit: np.ndarray
    Z_val: np.ndarray
    y_val: np.ndarray
    Z_test: np.ndarray
    y_test: np.ndarray
    latent_dim: int
    feature_set: str = "base"
    encoder_id: str = "autoencoder"
    y_train_all: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.Z_train_legit = np.asarray(self.Z_train_legit, dtype=np.float64)
        self.Z_val = np.asarray(self.Z_val, dtype=np.float64)
        self.Z_test = np.asarray(self.Z_test, dtype=np.float64)
        self.y_val = np.asarray(self.y_val, dtype=np.int32)
        self.y_test = np.asarray(self.y_test, dtype=np.int32)

        for name, arr in [
            ("Z_train_legit", self.Z_train_legit),
            ("Z_val", self.Z_val),
            ("Z_test", self.Z_test),
        ]:
            if arr.ndim != 2:
                raise ValueError(f"{name} must be 2-D, got shape {arr.shape}.")
            if arr.shape[1] != self.latent_dim:
                raise ValueError(
                    f"{name} column count {arr.shape[1]} != latent_dim={self.latent_dim}."
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN or Inf values.")

        if len(self.Z_val) != len(self.y_val):
            raise ValueError(
                f"Z_val length ({len(self.Z_val)}) != y_val length ({len(self.y_val)})."
            )
        if len(self.Z_test) != len(self.y_test):
            raise ValueError(
                f"Z_test length ({len(self.Z_test)}) != y_test length ({len(self.y_test)})."
            )
        if self.y_train_all is not None:
            self.y_train_all = np.asarray(self.y_train_all, dtype=np.int32)
            n_expected_legit = int((self.y_train_all == 0).sum())
            if len(self.Z_train_legit) != n_expected_legit:
                raise ValueError(
                    f"[LEAKAGE GUARD] Z_train_legit has {len(self.Z_train_legit)} rows "
                    f"but expected {n_expected_legit} legitimate training rows."
                )


# =============================================================================
# 1. Upstream Dependency Checks & Encoder Interface (Vani P2_03)
# =============================================================================


def discover_ae_checkpoints(models_dir: Path | None = None) -> list[Path]:
    """Return sorted trained Autoencoder checkpoint paths (``ae_*.pt``)."""
    target_dir = models_dir or (_ROOT / CFG["paths"]["models"])
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("ae_*.pt"))


def discover_ablation_checkpoints(models_dir: Path | None = None) -> dict[int, Path]:
    """Discover Vani's latent-dim ablation checkpoints (``ae_latent_k<k>.pt``)."""
    target_dir = models_dir or (_ROOT / CFG["paths"]["models"])
    if not target_dir.exists():
        return {}
    found: dict[int, Path] = {}
    for path in sorted(target_dir.glob("ae_latent_k*.pt")):
        stem = path.stem  # e.g. ae_latent_k8
        suffix = stem.replace("ae_latent_k", "")
        if suffix.isdigit():
            found[int(suffix)] = path
    return dict(sorted(found.items()))


def check_ae_dependencies(
    checkpoint_path: Path | None = None,
    models_dir: Path | None = None,
) -> Path:
    """
    Verify Vani's autoencoder module and trained weights exist.

    Raises
    ------
    WaitingForInputsError
        With an explicit ``waiting for inputs`` message when ``src/autoencoder.py``,
        its ``encode()`` interface, or ``results/models/ae_*.pt`` is absent.
    """
    ae_module_path = _ROOT / "src" / "autoencoder.py"
    if not ae_module_path.exists():
        raise WaitingForInputsError(
            "Waiting for inputs from Vani (P2_03): 'src/autoencoder.py' "
            "(exposing encode()) is not yet present in the repository."
        )

    try:
        ae_mod = importlib.import_module("src.autoencoder")
    except Exception as exc:
        raise WaitingForInputsError(
            f"Waiting for inputs from Vani (P2_03): failed to import 'src.autoencoder' ({exc})."
        ) from exc

    has_encode_class = any(
        hasattr(getattr(ae_mod, attr), "encode")
        for attr in dir(ae_mod)
        if not attr.startswith("_")
    )
    if not has_encode_class:
        raise WaitingForInputsError(
            "Waiting for inputs from Vani (P2_03): 'src/autoencoder.py' does not "
            "expose an encoder with an encode() method."
        )

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise WaitingForInputsError(
                f"Waiting for inputs from Vani (P2_03): checkpoint '{ckpt}' not found."
            )
        return ckpt

    ckpts = discover_ae_checkpoints(models_dir=models_dir)
    if not ckpts:
        target_dir = models_dir or (_ROOT / CFG["paths"]["models"])
        raise WaitingForInputsError(
            f"Waiting for inputs from Vani (P2_03): no trained autoencoder checkpoints "
            f"('ae_*.pt') found under '{target_dir}'."
        )
    return ckpts[0]


def _select_default_ae_checkpoint(ckpts: Sequence[Path]) -> Path:
    """Prefer the canonical undercomplete AE checkpoint if multiple exist."""
    for preferred in ("ae_best.pt", "ae_undercomplete.pt", "ae_final.pt", "ae.pt"):
        for ckpt in ckpts:
            if ckpt.name == preferred:
                return ckpt
    non_dae = [c for c in ckpts if "dae" not in c.name.lower()]
    return non_dae[0] if non_dae else ckpts[0]


def load_vani_encoder(
    checkpoint_path: Path | None = None,
    models_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Load Vani's trained Autoencoder model for inference-only encoding.

    Supports checkpoints saved either as a pickled model instance, a checkpoint
    dictionary containing config/hparams + state_dict, or via an explicit
    ``load_autoencoder`` helper if provided in ``src/autoencoder.py``.
    """
    if checkpoint_path is None:
        check_ae_dependencies(checkpoint_path=None, models_dir=models_dir)
        ckpt_path = _select_default_ae_checkpoint(discover_ae_checkpoints(models_dir))
    else:
        ckpt_path = check_ae_dependencies(checkpoint_path=checkpoint_path, models_dir=models_dir)

    import torch

    ae_mod = importlib.import_module("src.autoencoder")
    if hasattr(ae_mod, "load_autoencoder") and callable(ae_mod.load_autoencoder):
        model = ae_mod.load_autoencoder(ckpt_path)
        meta = {"checkpoint_path": str(ckpt_path), "loader": "src.autoencoder.load_autoencoder"}
        return model, meta

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if hasattr(payload, "encode") and callable(payload.encode):
        if hasattr(payload, "eval"):
            payload.eval()
        return payload, {"checkpoint_path": str(ckpt_path), "loader": "direct_module"}

    if not isinstance(payload, Mapping):
        raise WaitingForInputsError(
            f"Unsupported checkpoint payload in '{ckpt_path}': expected model or state dict."
        )

    state_dict = payload.get("model_state_dict") or payload.get("state_dict")
    if state_dict is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        state_dict = dict(payload)
    if state_dict is None:
        raise WaitingForInputsError(
            f"Checkpoint '{ckpt_path}' does not contain a recognizable state_dict."
        )

    hparams = dict(payload.get("hparams") or payload.get("config") or payload.get("model_kwargs") or {})
    model_cls_name = payload.get("model_class") or payload.get("arch")
    if model_cls_name and hasattr(ae_mod, str(model_cls_name)):
        cls = getattr(ae_mod, str(model_cls_name))
    elif "dae" in ckpt_path.name.lower() and hasattr(ae_mod, "DenoisingAE"):
        cls = ae_mod.DenoisingAE
    elif hasattr(ae_mod, "UndercompleteAE"):
        cls = ae_mod.UndercompleteAE
    else:
        raise WaitingForInputsError(
            "Could not locate UndercompleteAE or DenoisingAE in 'src/autoencoder.py'."
        )

    model = cls(**hparams)
    model.load_state_dict(state_dict)
    if hasattr(model, "eval"):
        model.eval()
    return model, {
        "checkpoint_path": str(ckpt_path),
        "hparams": hparams,
        "loader": cls.__name__,
    }


def encode_matrix(
    encoder: Any,
    X: np.ndarray,
    batch_size: int = 2048,
    device: str | None = None,
) -> np.ndarray:
    """
    Encode a tabular feature matrix ``X`` sequentially (no shuffling) via ``encoder.encode()``.

    Accepts either a PyTorch ``nn.Module`` or a lightweight Python stub implementing
    ``.encode(x)`` returning a ``torch.Tensor`` or ``np.ndarray``.
    """
    if not hasattr(encoder, "encode") or not callable(encoder.encode):
        raise TypeError("Encoder object must expose a callable .encode(x) method.")

    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2-D input matrix, got shape {X_arr.shape}.")

    is_torch_module = False
    try:
        import torch

        is_torch_module = isinstance(encoder, torch.nn.Module)
    except ImportError:
        torch = None  # type: ignore[assignment]

    if is_torch_module and torch is not None:
        target_device = (
            torch.device(device)
            if device is not None
            else next(encoder.parameters(), torch.empty(0)).device
        )
        encoder.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_arr), batch_size):
                batch = torch.from_numpy(X_arr[start : start + batch_size]).to(target_device)
                z_batch = encoder.encode(batch)
                if isinstance(z_batch, torch.Tensor):
                    z_np = z_batch.detach().cpu().numpy()
                else:
                    z_np = np.asarray(z_batch)
                chunks.append(z_np.astype(np.float64, copy=False))
        Z = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float64)
    else:
        chunks = []
        for start in range(0, len(X_arr), batch_size):
            z_batch = encoder.encode(X_arr[start : start + batch_size])
            if hasattr(z_batch, "detach"):
                z_np = z_batch.detach().cpu().numpy()
            else:
                z_np = np.asarray(z_batch)
            chunks.append(z_np.astype(np.float64, copy=False))
        Z = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float64)

    if len(Z) != len(X_arr):
        raise RuntimeError(
            f"Encoder output length {len(Z)} does not match input length {len(X_arr)}."
        )
    if not np.isfinite(Z).all():
        raise ValueError("Encoded latent matrix contains NaN or Inf values.")
    return Z


def encode_splits(
    encoder: Any,
    splits: Splits,
    batch_size: int = 2048,
    device: str | None = None,
    feature_set: str = "base",
    encoder_id: str = "autoencoder",
) -> LatentSplits:
    """
    Encode ``splits.X_train_legit``, ``splits.X_val``, and ``splits.X_test``
    while preserving chronological split alignment and zero-fraud training isolation.
    """
    Z_train_legit = encode_matrix(encoder, splits.X_train_legit, batch_size=batch_size, device=device)
    Z_val = encode_matrix(encoder, splits.X_val, batch_size=batch_size, device=device)
    Z_test = encode_matrix(encoder, splits.X_test, batch_size=batch_size, device=device)

    return LatentSplits(
        Z_train_legit=Z_train_legit,
        Z_val=Z_val,
        y_val=splits.y_val,
        Z_test=Z_test,
        y_test=splits.y_test,
        latent_dim=int(Z_train_legit.shape[1]),
        feature_set=feature_set,
        encoder_id=encoder_id,
        y_train_all=splits.y_train_all,
    )


# =============================================================================
# 2. Latent GMM Fitting, Scoring & Condition-Number Helpers
# =============================================================================


def _mean_condition_number(gmm: GaussianMixture) -> float:
    """Compute mean covariance matrix condition number across mixture components."""
    cov = gmm.covariance_type
    if cov == "full":
        conds = [float(np.linalg.cond(c)) for c in gmm.covariances_]
    elif cov == "tied":
        conds = [float(np.linalg.cond(gmm.covariances_))]
    elif cov == "diag":
        conds = [
            float(np.max(c) / max(float(np.min(c)), 1e-15))
            for c in gmm.covariances_
        ]
    elif cov == "spherical":
        conds = [1.0 for _ in gmm.covariances_]
    else:
        raise ValueError(f"Unsupported covariance_type='{cov}'.")
    return float(np.mean(conds))


def fit_latent_gmm(
    Z_train_legit: np.ndarray,
    n_components: int = 8,
    covariance_type: str = "full",
    reg_covar: float = 1e-4,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 3,
) -> GaussianMixture:
    """
    Fit a GMM exclusively on legitimate latent training vectors ``Z_train_legit``.

    Reuses :func:`src.gmm_model.fit_gmm` after validating finiteness and shape.
    """
    Z_arr = np.asarray(Z_train_legit, dtype=np.float64)
    if Z_arr.ndim != 2 or len(Z_arr) == 0:
        raise ValueError(f"Z_train_legit must be a non-empty 2-D array, got {Z_arr.shape}.")
    if not np.isfinite(Z_arr).all():
        raise ValueError("Z_train_legit contains NaN or Inf values.")

    return fit_gmm(
        X_train_legit=Z_arr,
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )


def compute_latent_nll(gmm: GaussianMixture, Z: np.ndarray) -> np.ndarray:
    """
    Compute latent Negative Log-Likelihood anomaly score ``-ln p(z)``.

    Higher values correspond to lower density in latent space (more anomalous).
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    scores = compute_anomaly_scores(gmm, Z_arr)
    if not np.isfinite(scores).all():
        raise ValueError("Computed latent GMM scores contain NaN or Inf values.")
    return scores


def _evaluate_gmm_once(
    Z_fit: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    covariance_type: str,
    n_components: int,
    reg_covar: float,
    seed: int,
    max_iter: int = 200,
    n_init: int = 1,
    Z_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one GMM configuration on ``Z_fit`` and evaluate on validation (and optional test)."""
    d = int(Z_fit.shape[1])
    p_theory = calculate_free_params(covariance_type, n_components, d)
    Z_val_legit = Z_val[y_val == 0]

    t0 = time.perf_counter()
    try:
        gmm = fit_latent_gmm(
            Z_fit,
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=seed,
            max_iter=max_iter,
            n_init=n_init,
        )
        fit_time = time.perf_counter() - t0
        train_aic = float(gmm.aic(Z_fit))
        train_bic = float(gmm.bic(Z_fit))
        val_legit_ll = float(gmm.score(Z_val_legit))
        n_vl = len(Z_val_legit)
        val_legit_bic = float(p_theory * np.log(n_vl) - 2.0 * n_vl * val_legit_ll)
        cond_num = _mean_condition_number(gmm)

        s_val = compute_latent_nll(gmm, Z_val)
        v_pr = float(pr_auc(y_val, s_val))
        v_roc = float(roc_auc(y_val, s_val))
        v_rec05 = float(recall_at_fpr(y_val, s_val, 0.005))

        out: dict[str, Any] = {
            "latent_dim": d,
            "covariance_type": covariance_type,
            "n_components": n_components,
            "reg_covar": reg_covar,
            "seed": seed,
            "free_parameters": p_theory,
            "train_aic": train_aic,
            "train_bic": train_bic,
            "val_legit_nll_mean": -val_legit_ll,
            "val_legit_bic": val_legit_bic,
            "val_pr_auc": v_pr,
            "val_roc_auc": v_roc,
            "val_recall@0.5%": v_rec05,
            "mean_condition_number": cond_num,
            "converged": bool(gmm.converged_),
            "n_iter": int(gmm.n_iter_),
            "train_time_sec": float(fit_time),
            "status": "Stable" if gmm.converged_ else "Unconverged",
        }
        if Z_test is not None and y_test is not None:
            s_test = compute_latent_nll(gmm, Z_test)
            out["test_pr_auc"] = float(pr_auc(y_test, s_test))
            out["test_roc_auc"] = float(roc_auc(y_test, s_test))
            out["test_recall@0.5%"] = float(recall_at_fpr(y_test, s_test, 0.005))
        return out
    except Exception as exc:
        fit_time = time.perf_counter() - t0
        out = {
            "latent_dim": d,
            "covariance_type": covariance_type,
            "n_components": n_components,
            "reg_covar": reg_covar,
            "seed": seed,
            "free_parameters": p_theory,
            "train_aic": np.nan,
            "train_bic": np.nan,
            "val_legit_nll_mean": np.nan,
            "val_legit_bic": np.nan,
            "val_pr_auc": np.nan,
            "val_roc_auc": np.nan,
            "val_recall@0.5%": np.nan,
            "mean_condition_number": np.nan,
            "converged": False,
            "n_iter": -1,
            "train_time_sec": float(fit_time),
            "status": f"Failed ({type(exc).__name__})",
        }
        if Z_test is not None and y_test is not None:
            out["test_pr_auc"] = np.nan
            out["test_roc_auc"] = np.nan
            out["test_recall@0.5%"] = np.nan
        return out


# =============================================================================
# 3. Task A — Latent GMM Grid Search, Selection & Parent Comparison
# =============================================================================


def run_latent_gmm_grid_search(
    latent_splits: LatentSplits,
    cov_types: Sequence[str] = ("full", "tied", "diag", "spherical"),
    k_values: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
    reg_covars: Sequence[float] = (1e-6, 1e-4, 1e-2),
    subsample_size: int | None = 50000,
    seed: int = 42,
    n_init: int = 1,
    max_iter: int = 200,
) -> pd.DataFrame:
    """
    Sweep covariance structure x component count K x regularization on latent codes.

    Uses train-legit codes only (optionally subsampled to ``subsample_size`` for
    screening) and evaluates AIC/BIC plus validation-only fraud triage metrics.
    """
    rng = np.random.default_rng(seed)
    n_train = len(latent_splits.Z_train_legit)
    if subsample_size is not None and subsample_size < n_train:
        sub_idx = rng.choice(n_train, size=subsample_size, replace=False)
        Z_sub = latent_splits.Z_train_legit[sub_idx]
    else:
        Z_sub = latent_splits.Z_train_legit

    _logger.info(
        "Running latent GMM grid search (d=%d, fit_rows=%d, configs=%d)...",
        latent_splits.latent_dim,
        len(Z_sub),
        len(cov_types) * len(k_values) * len(reg_covars),
    )

    rows = [
        _evaluate_gmm_once(
            Z_fit=Z_sub,
            Z_val=latent_splits.Z_val,
            y_val=latent_splits.y_val,
            covariance_type=cov,
            n_components=K,
            reg_covar=reg,
            seed=seed,
            max_iter=max_iter,
            n_init=n_init,
        )
        for cov in cov_types
        for reg in reg_covars
        for K in k_values
    ]
    return pd.DataFrame(rows)


def select_latent_gmm_config(
    grid_df: pd.DataFrame,
    selection_criterion: str = "val_pr_auc",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Compare unsupervised BIC selection vs. validation PR-AUC selection (no test data).

    Parameters
    ----------
    grid_df : pd.DataFrame
        Output of :func:`run_latent_gmm_grid_search`.
    selection_criterion : {'val_pr_auc', 'train_bic', 'val_legit_bic'}
        Pre-committed selection criterion for choosing the final H1 config.
    """
    valid_df = grid_df.dropna(subset=["train_bic", "val_pr_auc"]).copy()
    if valid_df.empty:
        raise RuntimeError("No valid converged GMM configurations in grid search.")

    row_train_bic = valid_df.loc[valid_df["train_bic"].idxmin()]
    row_val_bic = valid_df.loc[valid_df["val_legit_bic"].idxmin()]
    row_val_pr = valid_df.loc[valid_df["val_pr_auc"].idxmax()]

    summary_rows = []
    for label, row in [
        ("Unsupervised (Min Train-Legit BIC)", row_train_bic),
        ("Unsupervised (Min Val-Legit BIC)", row_val_bic),
        ("Validation-Selected (Max Val PR-AUC)", row_val_pr),
    ]:
        summary_rows.append(
            {
                "criterion": label,
                "latent_dim": int(row["latent_dim"]),
                "covariance_type": str(row["covariance_type"]),
                "n_components": int(row["n_components"]),
                "reg_covar": float(row["reg_covar"]),
                "free_parameters": int(row["free_parameters"]),
                "train_bic": round(float(row["train_bic"]), 2),
                "val_legit_bic": round(float(row["val_legit_bic"]), 2),
                "val_pr_auc": round(float(row["val_pr_auc"]), 5),
                "val_recall@0.5%": round(float(row["val_recall@0.5%"]), 5),
            }
        )
    comparison_df = pd.DataFrame(summary_rows)

    selector_map = {
        "val_pr_auc": row_val_pr,
        "train_bic": row_train_bic,
        "val_legit_bic": row_val_bic,
    }
    if selection_criterion not in selector_map:
        raise ValueError(f"Unknown selection_criterion='{selection_criterion}'.")

    chosen = selector_map[selection_criterion]
    chosen_cfg = {
        "selection_criterion": selection_criterion,
        "latent_dim": int(chosen["latent_dim"]),
        "covariance_type": str(chosen["covariance_type"]),
        "n_components": int(chosen["n_components"]),
        "reg_covar": float(chosen["reg_covar"]),
        "free_parameters": int(chosen["free_parameters"]),
        "grid_train_bic": float(chosen["train_bic"]),
        "grid_val_pr_auc": float(chosen["val_pr_auc"]),
    }
    return chosen_cfg, comparison_df


def compare_with_parents(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    parent_scores: Mapping[str, Mapping[str, np.ndarray]],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare H1 Latent-GMM against parent models (Phase 1 GMM, AE, DAE, PCA)
    using stratified bootstrap CIs and paired bootstrap difference tests.
    """
    all_models: dict[str, Mapping[str, np.ndarray]] = {
        "hybrid_latentgmm": {"val": s_val, "test": s_test},
    }
    all_models.update(parent_scores)

    ci_rows = []
    for name, scores_dict in all_models.items():
        sv = np.asarray(scores_dict["val"], dtype=np.float64)
        st = np.asarray(scores_dict["test"], dtype=np.float64)
        ci_pr = bootstrap_ci(pr_auc, y_test, st, n=n_boot, seed=seed)
        ci_roc = bootstrap_ci(roc_auc, y_test, st, n=n_boot, seed=seed)
        ci_rec = bootstrap_ci(recall_at_fpr, y_test, st, n=n_boot, seed=seed, fpr=0.005)
        ci_rows.append(
            {
                "model": name,
                "PR-AUC_val": float(pr_auc(y_val, sv)),
                "PR-AUC_test": ci_pr["point"],
                "PR-AUC_test_ci_lo": ci_pr["ci_lo"],
                "PR-AUC_test_ci_hi": ci_pr["ci_hi"],
                "ROC-AUC_test": ci_roc["point"],
                "recall@FPR=0.5%_test": ci_rec["point"],
                "recall@FPR=0.5%_test_ci_lo": ci_rec["ci_lo"],
                "recall@FPR=0.5%_test_ci_hi": ci_rec["ci_hi"],
            }
        )

    paired_rows = []
    for name, scores_dict in parent_scores.items():
        st_parent = np.asarray(scores_dict["test"], dtype=np.float64)
        diff_pr = paired_bootstrap_diff(
            y_test, s_test, st_parent, pr_auc, n=n_boot, seed=seed
        )
        diff_rec = paired_bootstrap_diff(
            y_test, s_test, st_parent, recall_at_fpr, n=n_boot, seed=seed, fpr=0.005
        )
        paired_rows.append(
            {
                "comparison": f"hybrid_latentgmm - {name}",
                "baseline_model": name,
                "delta_pr_auc_test": diff_pr["diff_point"],
                "pr_auc_ci_lo": diff_pr["ci_lo"],
                "pr_auc_ci_hi": diff_pr["ci_hi"],
                "pr_auc_significant": bool(diff_pr["significant"]),
                "delta_recall@0.5%_test": diff_rec["diff_point"],
                "recall_ci_lo": diff_rec["ci_lo"],
                "recall_ci_hi": diff_rec["ci_hi"],
                "recall_significant": bool(diff_rec["significant"]),
            }
        )

    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)


# =============================================================================
# 4. Task B — Why Latent-GMM Works (Evidence Benchmarks i–v)
# =============================================================================


def compute_parameter_count_comparison(
    ambient_dims: Sequence[int] = (30, 31),
    latent_dims: Sequence[int] = (2, 4, 8, 16),
    k_values: Sequence[int] = (1, 2, 4, 5, 8, 12, 16),
    cov_types: Sequence[str] = ("full", "tied", "diag", "spherical"),
) -> pd.DataFrame:
    """
    Task B(i): Exact GMM free-parameter arithmetic comparing ambient spaces
    (d=30 raw, d=31 base/engineered) against latent dimensions (e.g. d=8).
    """
    rows = []
    all_dims = [(d, "ambient") for d in ambient_dims] + [(d, "latent") for d in latent_dims]
    ref_dim = max(ambient_dims)

    for d, space in all_dims:
        for cov in cov_types:
            for K in k_values:
                p = calculate_free_params(cov, K, d)
                p_ref = calculate_free_params(cov, K, ref_dim)
                rows.append(
                    {
                        "space": space,
                        "dimension": int(d),
                        "covariance_type": cov,
                        "n_components": int(K),
                        "free_parameters": int(p),
                        "cov_params_only": int(
                            K * d * (d + 1) // 2
                            if cov == "full"
                            else d * (d + 1) // 2
                            if cov == "tied"
                            else K * d
                            if cov == "diag"
                            else K
                        ),
                        f"ratio_vs_d{ref_dim}": round(p / p_ref, 4),
                        f"reduction_pct_vs_d{ref_dim}": round(100.0 * (1.0 - p / p_ref), 2),
                    }
                )
    return pd.DataFrame(rows)


def run_reg_covar_sensitivity(
    X_train_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    Z_train_legit: np.ndarray,
    Z_val: np.ndarray,
    n_components: int = 8,
    reg_values: Sequence[float] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    cov_types: Sequence[str] = ("full", "diag"),
    subsample_size: int = 30000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Task B(ii): Compare covariance condition numbers, stability, and validation
    PR-AUC across ``reg_covar`` values on ambient features vs. latent codes.
    """
    rng = np.random.default_rng(seed)
    n_rows = min(subsample_size, len(X_train_legit), len(Z_train_legit))
    idx = rng.choice(len(X_train_legit), size=n_rows, replace=False)

    X_sub = np.asarray(X_train_legit[idx], dtype=np.float64)
    Z_sub = np.asarray(Z_train_legit[idx], dtype=np.float64)

    rows = []
    for space_label, train_mat, val_mat in [
        (f"ambient_d{X_train_legit.shape[1]}", X_sub, np.asarray(X_val, dtype=np.float64)),
        (f"latent_d{Z_train_legit.shape[1]}", Z_sub, np.asarray(Z_val, dtype=np.float64)),
    ]:
        for cov in cov_types:
            for reg in reg_values:
                res = _evaluate_gmm_once(
                    Z_fit=train_mat,
                    Z_val=val_mat,
                    y_val=y_val,
                    covariance_type=cov,
                    n_components=n_components,
                    reg_covar=reg,
                    seed=seed,
                    max_iter=200,
                    n_init=1,
                )
                res["space"] = space_label
                rows.append(res)
    return pd.DataFrame(rows)


def run_seed_stability_comparison(
    X_train_legit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    Z_train_legit: np.ndarray,
    Z_val: np.ndarray,
    ambient_cfg: Mapping[str, Any] | None = None,
    latent_cfg: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    subsample_size: int | None = None,
) -> pd.DataFrame:
    """
    Task B(iii): Measure initialization-seed variance of validation PR-AUC for
    ambient-space GMM vs. latent-space GMM across multiple seeds (``n_init=1``).
    """
    amb_params = {"covariance_type": "full", "n_components": 8, "reg_covar": 1e-2}
    if ambient_cfg:
        amb_params.update(ambient_cfg)
    lat_params = {"covariance_type": "full", "n_components": 8, "reg_covar": 1e-4}
    if latent_cfg:
        lat_params.update(latent_cfg)

    rng = np.random.default_rng(42)
    if subsample_size is not None and subsample_size < len(X_train_legit):
        idx = rng.choice(len(X_train_legit), size=subsample_size, replace=False)
        X_fit = np.asarray(X_train_legit[idx], dtype=np.float64)
        Z_fit = np.asarray(Z_train_legit[idx], dtype=np.float64)
    else:
        X_fit = np.asarray(X_train_legit, dtype=np.float64)
        Z_fit = np.asarray(Z_train_legit, dtype=np.float64)

    rows = []
    for space_label, train_mat, val_mat, params in [
        (f"ambient_d{X_train_legit.shape[1]}", X_fit, np.asarray(X_val, dtype=np.float64), amb_params),
        (f"latent_d{Z_train_legit.shape[1]}", Z_fit, np.asarray(Z_val, dtype=np.float64), lat_params),
    ]:
        for seed in seeds:
            res = _evaluate_gmm_once(
                Z_fit=train_mat,
                Z_val=val_mat,
                y_val=y_val,
                covariance_type=str(params["covariance_type"]),
                n_components=int(params["n_components"]),
                reg_covar=float(params["reg_covar"]),
                seed=int(seed),
                max_iter=200,
                n_init=1,
            )
            res["space"] = space_label
            rows.append(res)
    return pd.DataFrame(rows)


def _compute_gaussianity_rows(
    matrix: np.ndarray,
    space_label: str,
    col_names: Sequence[str],
    sample_size: int = 5000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compute skewness, excess kurtosis, and Shapiro-Wilk normality stats per coordinate."""
    arr = np.asarray(matrix, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(arr) > sample_size:
        idx = rng.choice(len(arr), size=sample_size, replace=False)
        shapiro_mat = arr[idx]
    else:
        shapiro_mat = arr

    rows = []
    for j, name in enumerate(col_names):
        col = arr[:, j]
        skew_val = float(stats.skew(col))
        kurt_val = float(stats.kurtosis(col))  # excess kurtosis (0 for Gaussian)
        if np.std(shapiro_mat[:, j]) > 1e-12:
            sw_stat, sw_p = stats.shapiro(shapiro_mat[:, j])
        else:
            sw_stat, sw_p = np.nan, np.nan
        rows.append(
            {
                "space": space_label,
                "coordinate": name,
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "skewness": skew_val,
                "excess_kurtosis": kurt_val,
                "abs_skew": abs(skew_val),
                "abs_kurtosis": abs(kurt_val),
                "mildly_gaussian": bool(abs(skew_val) < 1.0 and abs(kurt_val) < 3.0),
                "shapiro_w": float(sw_stat) if not np.isnan(sw_stat) else np.nan,
                "shapiro_p": float(sw_p) if not np.isnan(sw_p) else np.nan,
            }
        )
    return rows


def analyze_latent_gaussianity(
    X_train_legit: np.ndarray,
    Z_train_legit: np.ndarray,
    feature_names: Sequence[str] | None = None,
    sample_size: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Task B(iv): Compare marginal skewness, excess kurtosis, and Shapiro-Wilk W
    between ambient training-legitimate coordinates and latent coordinates.
    """
    d_x = X_train_legit.shape[1]
    d_z = Z_train_legit.shape[1]
    x_names = list(feature_names) if feature_names is not None else [f"x_{i}" for i in range(d_x)]
    z_names = [f"z_{i+1}" for i in range(d_z)]

    rows = _compute_gaussianity_rows(
        X_train_legit, f"ambient_d{d_x}", x_names, sample_size=sample_size, seed=seed
    )
    rows.extend(
        _compute_gaussianity_rows(
            Z_train_legit, f"latent_d{d_z}", z_names, sample_size=sample_size, seed=seed
        )
    )
    return pd.DataFrame(rows)


def summarize_gaussianity(gauss_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-coordinate normality diagnostics by space (ambient vs. latent)."""
    return (
        gauss_df.groupby("space", as_index=False)
        .agg(
            n_coords=("coordinate", "count"),
            mean_abs_skew=("abs_skew", "mean"),
            median_abs_skew=("abs_skew", "median"),
            max_abs_skew=("abs_skew", "max"),
            mean_abs_kurtosis=("abs_kurtosis", "mean"),
            median_abs_kurtosis=("abs_kurtosis", "median"),
            max_abs_kurtosis=("abs_kurtosis", "max"),
            pct_mildly_gaussian=("mildly_gaussian", lambda s: float(np.mean(s) * 100.0)),
            mean_shapiro_w=("shapiro_w", "mean"),
        )
    )


def compute_complementarity_summary(
    y_split: np.ndarray,
    s_latent: np.ndarray,
    parent_scores: Mapping[str, np.ndarray],
    fpr_budgets: Sequence[float] = (0.001, 0.005, 0.01),
) -> pd.DataFrame:
    """
    Quantify alert complementarity (H1-only, parent-only, both, missed-by-both)
    among true fraud cases at fixed FPR operating budgets.
    """
    y_arr = np.asarray(y_split, dtype=np.int32)
    s_h1 = np.asarray(s_latent, dtype=np.float64)
    fraud_mask = y_arr == 1
    legit_mask = y_arr == 0
    n_fraud = int(fraud_mask.sum())
    if n_fraud == 0:
        return pd.DataFrame()

    rows = []
    for fpr in fpr_budgets:
        th_h1 = float(np.quantile(s_h1[legit_mask], 1.0 - fpr))
        caught_h1 = (s_h1 >= th_h1) & fraud_mask
        for name, p_scores in parent_scores.items():
            s_p = np.asarray(p_scores, dtype=np.float64)
            th_p = float(np.quantile(s_p[legit_mask], 1.0 - fpr))
            caught_p = (s_p >= th_p) & fraud_mask
            both = int((caught_h1 & caught_p).sum())
            h1_only = int((caught_h1 & ~caught_p).sum())
            parent_only = int((~caught_h1 & caught_p).sum())
            neither = int((~caught_h1 & ~caught_p & fraud_mask).sum())
            rows.append(
                {
                    "parent_model": name,
                    "fpr_budget": float(fpr),
                    "n_fraud": n_fraud,
                    "h1_caught": int(caught_h1.sum()),
                    "parent_caught": int(caught_p.sum()),
                    "both_caught": both,
                    "h1_only_caught": h1_only,
                    "parent_only_caught": parent_only,
                    "missed_by_both": neither,
                    "union_recall": float((both + h1_only + parent_only) / n_fraud),
                }
            )
    return pd.DataFrame(rows)


def analyze_latentgmm_failures(
    y_split: np.ndarray,
    s_latent: np.ndarray,
    parent_scores: Mapping[str, np.ndarray] | None = None,
    Z_split: np.ndarray | None = None,
    n_missed: int = 10,
    fpr_budget: float = 0.005,
) -> pd.DataFrame:
    """
    Task B(v): Honest failure analysis inspecting the lowest-scored frauds
    under Latent-GMM alongside parent-model percentiles at a fixed FPR budget.
    """
    y_arr = np.asarray(y_split, dtype=np.int32)
    s_arr = np.asarray(s_latent, dtype=np.float64)
    fraud_idx = np.where(y_arr == 1)[0]
    if len(fraud_idx) == 0:
        return pd.DataFrame()

    legit_scores = s_arr[y_arr == 0]
    thresh_h1 = float(np.quantile(legit_scores, 1.0 - fpr_budget)) if len(legit_scores) else np.inf
    order_missed = fraud_idx[np.argsort(s_arr[fraud_idx])[:n_missed]]

    parent_thresholds: dict[str, float] = {}
    if parent_scores:
        for name, p_scores in parent_scores.items():
            p_arr = np.asarray(p_scores, dtype=np.float64)
            p_legit = p_arr[y_arr == 0]
            parent_thresholds[name] = (
                float(np.quantile(p_legit, 1.0 - fpr_budget)) if len(p_legit) else np.inf
            )

    rows = []
    for rank, idx in enumerate(order_missed, start=1):
        row: dict[str, Any] = {
            "missed_rank": rank,
            "row_index": int(idx),
            "latentgmm_nll": round(float(s_arr[idx]), 4),
            "latentgmm_legit_pct": round(float((legit_scores <= s_arr[idx]).mean() * 100.0), 2),
            "flagged_at_fpr_budget": bool(s_arr[idx] >= thresh_h1),
        }
        if Z_split is not None:
            row["latent_l2_norm"] = round(float(np.linalg.norm(Z_split[idx])), 4)
        if parent_scores:
            for name, p_scores in parent_scores.items():
                val = float(p_scores[idx])
                row[f"{name}_score"] = round(val, 4)
                row[f"{name}_flagged"] = bool(val >= parent_thresholds[name])
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 5. Task C — Multi-Seed GMM Ablation Grid (Feeds Prachee's Master Ablation)
# =============================================================================


def run_gmm_ablation(
    latent_splits_by_dim: Mapping[int, LatentSplits],
    cov_types: Sequence[str] = ("full", "tied", "diag", "spherical"),
    k_values: Sequence[int] = (1, 2, 4, 8, 12, 16),
    reg_covar: float = 1e-4,
    seeds: Sequence[int] = (42, 43, 44),
    subsample_size: int | None = 50000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task C: Execute ``latent_dim x cov_type x K`` ablation over 3 seeds.

    Reports BOTH ``val_pr_auc`` and ``test_pr_auc`` across seeds. Note:
    ``test_pr_auc`` is logged purely for ablation reporting and is NEVER used
    for hyperparameter selection.
    """
    run_rows = []
    for d_lat in sorted(latent_splits_by_dim.keys()):
        ls = latent_splits_by_dim[d_lat]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            n_tr = len(ls.Z_train_legit)
            if subsample_size is not None and subsample_size < n_tr:
                idx = rng.choice(n_tr, size=subsample_size, replace=False)
                Z_fit = ls.Z_train_legit[idx]
            else:
                Z_fit = ls.Z_train_legit

            for cov in cov_types:
                for K in k_values:
                    res = _evaluate_gmm_once(
                        Z_fit=Z_fit,
                        Z_val=ls.Z_val,
                        y_val=ls.y_val,
                        covariance_type=cov,
                        n_components=K,
                        reg_covar=reg_covar,
                        seed=int(seed),
                        max_iter=200,
                        n_init=1,
                        Z_test=ls.Z_test,
                        y_test=ls.y_test,
                    )
                    res["feature_set"] = ls.feature_set
                    run_rows.append(res)

    runs_df = pd.DataFrame(run_rows)
    summary_df = (
        runs_df.groupby(["latent_dim", "covariance_type", "n_components", "reg_covar"], as_index=False)
        .agg(
            free_parameters=("free_parameters", "first"),
            n_seeds=("seed", "count"),
            n_converged=("converged", "sum"),
            val_pr_auc_mean=("val_pr_auc", "mean"),
            val_pr_auc_std=("val_pr_auc", "std"),
            test_pr_auc_mean=("test_pr_auc", "mean"),
            test_pr_auc_std=("test_pr_auc", "std"),
            val_recall05_mean=("val_recall@0.5%", "mean"),
            test_recall05_mean=("test_recall@0.5%", "mean"),
            train_bic_mean=("train_bic", "mean"),
            train_time_mean_sec=("train_time_sec", "mean"),
        )
        .sort_values(["latent_dim", "val_pr_auc_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return runs_df, summary_df


# =============================================================================
# 6. Visualizations (saved to results/figures/latentgmm_*.png)
# =============================================================================


def plot_latent_aic_bic(grid_df: pd.DataFrame, default_reg: float = 1e-4) -> plt.Figure:
    """Plot AIC and BIC curves vs. K on latent codes for each covariance structure."""
    avail_regs = sorted(grid_df["reg_covar"].dropna().unique())
    reg_use = default_reg if default_reg in avail_regs else avail_regs[0]
    sub_df = grid_df[grid_df["reg_covar"] == reg_use].copy()

    cov_types = ["full", "tied", "diag", "spherical"]
    colors = {"full": "#D0021B", "tied": "#F5A623", "diag": "#4A90E2", "spherical": "#7ED321"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for cov in cov_types:
        df_c = sub_df[sub_df["covariance_type"] == cov].sort_values("n_components")
        if df_c.empty:
            continue
        ax1.plot(df_c["n_components"], df_c["train_aic"], marker="o", label=cov.capitalize(), color=colors[cov])
        ax2.plot(df_c["n_components"], df_c["train_bic"], marker="s", label=cov.capitalize(), color=colors[cov])

    if not sub_df["train_bic"].dropna().empty:
        min_bic = sub_df.loc[sub_df["train_bic"].idxmin()]
        ax2.scatter(
            [min_bic["n_components"]],
            [min_bic["train_bic"]],
            color="black",
            s=120,
            zorder=5,
            marker="*",
            label=f"Min BIC: {min_bic['covariance_type']}, K={int(min_bic['n_components'])}",
        )

    ax1.set_title(f"Latent GMM AIC vs. K (reg={reg_use})")
    ax1.set_xlabel("Components (K)")
    ax1.set_ylabel("Train-Legit AIC (Lower is Better)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Latent GMM BIC vs. K (reg={reg_use})")
    ax2.set_xlabel("Components (K)")
    ax2.set_ylabel("Train-Legit BIC (Lower is Better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_latent_nll_distribution(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
) -> plt.Figure:
    """Plot log-scale histogram of latent GMM NLL scores for legit vs. fraud."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y, s, title in [
        (ax1, y_val, s_val, "Latent-GMM NLL Distribution (Validation)"),
        (ax2, y_test, s_test, "Latent-GMM NLL Distribution (Test)"),
    ]:
        ax.hist(s[y == 0], bins=80, density=True, alpha=0.55, color="#1F77B4", label="Legitimate")
        ax.hist(s[y == 1], bins=40, density=True, alpha=0.70, color="#D62728", label="Fraud")
        ax.set_yscale("log")
        ax.set_xlabel("Latent Negative Log-Likelihood (-ln p(z))")
        ax.set_ylabel("Density (Log Scale)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_latent_stability_evidence(
    reg_df: pd.DataFrame,
    seed_df: pd.DataFrame,
) -> plt.Figure:
    """Plot condition number vs. reg_covar and seed PR-AUC variance (ambient vs. latent)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    full_reg = reg_df[reg_df["covariance_type"] == "full"]
    for space, grp in full_reg.groupby("space"):
        grp_s = grp.sort_values("reg_covar")
        ax1.plot(grp_s["reg_covar"], grp_s["mean_condition_number"], marker="o", label=f"{space} (full)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Covariance Regularization (reg_covar)")
    ax1.set_ylabel("Mean Condition Number (log scale)")
    ax1.set_title("Ill-Conditioning: Ambient vs. Latent Full Covariance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    spaces = sorted(seed_df["space"].unique())
    data_boxes = [seed_df.loc[seed_df["space"] == sp, "val_pr_auc"].dropna().values for sp in spaces]
    ax2.boxplot(data_boxes, tick_labels=spaces, patch_artist=True)
    ax2.set_ylabel("Validation PR-AUC across Seeds")
    ax2.set_title("EM Initialization Seed Variance: Ambient vs. Latent")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_latent_qq(Z_train_legit: np.ndarray, max_dims: int = 8) -> plt.Figure:
    """Plot Normal QQ-plots for up to ``max_dims`` latent dimensions on a seeded subsample."""
    d_plot = min(Z_train_legit.shape[1], max_dims)
    rng = np.random.default_rng(42)
    n_sub = min(5000, len(Z_train_legit))
    sub = Z_train_legit[rng.choice(len(Z_train_legit), size=n_sub, replace=False)]

    n_cols = 4 if d_plot >= 4 else d_plot
    n_rows = int(np.ceil(d_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for j in range(d_plot):
        stats.probplot(sub[:, j], dist="norm", plot=axes_flat[j])
        axes_flat[j].set_title(f"Latent z_{j+1} Normal QQ")
    for k in range(d_plot, len(axes_flat)):
        axes_flat[k].axis("off")

    plt.tight_layout()
    return fig


def plot_latent_2d_projection(
    Z_val: np.ndarray,
    y_val: np.ndarray,
    max_legit_points: int = 5000,
    seed: int = 42,
) -> plt.Figure:
    """Plot 2-D visualization of validation latent codes (raw if d=2 else PCA projection)."""
    if Z_val.shape[1] == 2:
        coords = Z_val
        x_lbl, y_lbl = "z_1", "z_2"
    else:
        pca = PCA(n_components=2, random_state=seed)
        legit_mask = y_val == 0
        pca.fit(Z_val[legit_mask])
        coords = pca.transform(Z_val)
        x_lbl, y_lbl = "Latent PC 1", "Latent PC 2"

    rng = np.random.default_rng(seed)
    legit_idx = np.where(y_val == 0)[0]
    if len(legit_idx) > max_legit_points:
        legit_idx = rng.choice(legit_idx, size=max_legit_points, replace=False)
    fraud_idx = np.where(y_val == 1)[0]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        coords[legit_idx, 0],
        coords[legit_idx, 1],
        s=8,
        alpha=0.25,
        color="#1F77B4",
        label=f"Legitimate (n={len(legit_idx)})",
    )
    ax.scatter(
        coords[fraud_idx, 0],
        coords[fraud_idx, 1],
        s=32,
        alpha=0.85,
        color="#D62728",
        marker="^",
        label=f"Fraud (n={len(fraud_idx)})",
    )
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    ax.set_title("Validation Latent Manifold Projection (Legit vs. Fraud)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# 7. In-Memory Synthetic Stub Self-Test (Never Writes to results/)
# =============================================================================


class _SyntheticStubEncoder:
    """Deterministic linear+tanh projection stub used strictly for offline tests."""

    def __init__(self, d_in: int, d_latent: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(d_in, d_latent)).astype(np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(x, dtype=np.float32) @ self.proj)


def run_synthetic_stub_self_test(seed: int = 42) -> dict[str, Any]:
    """
    Validate the entire Latent-GMM pipeline end-to-end on labelled synthetic
    stubs strictly in memory (zero files created under ``results/``).
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    d_in, d_latent = 31, 8
    n_train_legit, n_val_legit, n_val_fraud = 1200, 500, 25
    n_test_legit, n_test_fraud = 500, 25

    X_tr_leg = rng.normal(0.0, 1.0, size=(n_train_legit, d_in)).astype(np.float32)
    X_val_leg = rng.normal(0.0, 1.0, size=(n_val_legit, d_in)).astype(np.float32)
    X_val_frd = rng.normal(2.2, 1.3, size=(n_val_fraud, d_in)).astype(np.float32)
    X_te_leg = rng.normal(0.0, 1.0, size=(n_test_legit, d_in)).astype(np.float32)
    X_te_frd = rng.normal(2.2, 1.3, size=(n_test_fraud, d_in)).astype(np.float32)

    X_val = np.vstack([X_val_leg, X_val_frd])
    y_val = np.array([0] * n_val_legit + [1] * n_val_fraud, dtype=np.int32)
    X_test = np.vstack([X_te_leg, X_te_frd])
    y_test = np.array([0] * n_test_legit + [1] * n_test_fraud, dtype=np.int32)

    splits = Splits(
        X_train_legit=X_tr_leg,
        X_train_all=X_tr_leg,
        y_train_all=np.zeros(n_train_legit, dtype=np.int32),
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=[f"f{i}" for i in range(d_in)],
        scaler=None,  # type: ignore[arg-type]
    )

    stub_enc = _SyntheticStubEncoder(d_in=d_in, d_latent=d_latent, seed=seed)
    ls = encode_splits(stub_enc, splits, batch_size=256, feature_set="stub", encoder_id="stub_ae")

    grid_df = run_latent_gmm_grid_search(
        ls,
        cov_types=("full", "diag"),
        k_values=(1, 2, 4),
        reg_covars=(1e-4, 1e-2),
        subsample_size=None,
        seed=seed,
    )
    chosen_cfg, sel_df = select_latent_gmm_config(grid_df, selection_criterion="val_pr_auc")
    gmm = fit_latent_gmm(
        ls.Z_train_legit,
        n_components=chosen_cfg["n_components"],
        covariance_type=chosen_cfg["covariance_type"],
        reg_covar=chosen_cfg["reg_covar"],
        random_state=seed,
    )
    s_val = compute_latent_nll(gmm, ls.Z_val)
    s_test = compute_latent_nll(gmm, ls.Z_test)

    assert s_val.dtype == np.float32 and s_val.shape == (len(y_val),)
    assert s_test.dtype == np.float32 and s_test.shape == (len(y_test),)
    assert float(s_test[y_test == 1].mean()) > float(s_test[y_test == 0].mean())

    val_pr = pr_auc(y_val, s_val)
    test_pr = pr_auc(y_test, s_test)
    _logger.info(
        "[STUB SELF-TEST PASS] d=%d->%d, chosen=%s K=%d, val_pr=%.4f, test_pr=%.4f",
        d_in,
        d_latent,
        chosen_cfg["covariance_type"],
        chosen_cfg["n_components"],
        val_pr,
        test_pr,
    )
    return {
        "chosen_cfg": chosen_cfg,
        "selection_df": sel_df,
        "val_pr_auc": val_pr,
        "test_pr_auc": test_pr,
    }


# =============================================================================
# 8. Real Phase 2 Pipeline Entry Point
# =============================================================================


def _load_available_parent_scores() -> dict[str, dict[str, np.ndarray]]:
    """Load available parent score arrays (Phase 1 GMM, AE, DAE, PCA) without crashing."""
    parents: dict[str, dict[str, np.ndarray]] = {}
    for model_name in ("gmm", "ae", "ae_dae", "pca"):
        try:
            parents[model_name] = {
                "val": load_scores(model_name, "val"),
                "test": load_scores(model_name, "test"),
            }
        except FileNotFoundError:
            _logger.warning("Optional parent score '%s' not found; skipping.", model_name)
    return parents


def run_day2_stage(
    feature_set: str = "base",
    checkpoint_path: Path | None = None,
    selection_criterion: str = "val_pr_auc",
    n_boot: int = 1000,
) -> dict[str, Any]:
    """
    Execute Day 2 H1 Latent-GMM experiments:
    - Task A: Encode via Vani's chosen AE, grid search cov x K x reg, select on val,
      refit on all train-legit codes, save ``hybrid_latentgmm_{val,test}.npy``,
      evaluate metrics, and run paired bootstrap comparison vs. parents.
    - Task B (i)-(iii): Parameter counts, reg_covar/condition-number sensitivity,
      and 5-seed initialization variance (ambient vs. latent).
    """
    seed = int(CFG.get("seed", 42))
    set_seed(seed)

    encoder, encoder_meta = load_vani_encoder(checkpoint_path=checkpoint_path)
    _logger.info("Loaded Vani encoder: %s", encoder_meta)

    splits = get_features(feature_set=feature_set)
    latent_splits = encode_splits(
        encoder,
        splits,
        feature_set=feature_set,
        encoder_id=Path(encoder_meta["checkpoint_path"]).stem,
    )

    grid_df = run_latent_gmm_grid_search(latent_splits, seed=seed)
    save_table(grid_df, "latentgmm_grid_search")

    chosen_cfg, sel_df = select_latent_gmm_config(grid_df, selection_criterion=selection_criterion)
    chosen_cfg["encoder_checkpoint"] = Path(encoder_meta["checkpoint_path"]).name
    chosen_cfg["feature_set"] = feature_set
    save_table(sel_df, "latentgmm_selection_comparison")
    save_table(pd.DataFrame([chosen_cfg]), "latentgmm_selected_config")

    gmm_final = fit_latent_gmm(
        latent_splits.Z_train_legit,
        n_components=chosen_cfg["n_components"],
        covariance_type=chosen_cfg["covariance_type"],
        reg_covar=chosen_cfg["reg_covar"],
        random_state=seed,
        max_iter=300,
        n_init=3,
    )
    models_dir = _ROOT / CFG["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gmm_final, models_dir / "hybrid_latentgmm.joblib")

    s_val = compute_latent_nll(gmm_final, latent_splits.Z_val)
    s_test = compute_latent_nll(gmm_final, latent_splits.Z_test)
    save_scores("hybrid_latentgmm", "val", s_val)
    save_scores("hybrid_latentgmm", "test", s_test)

    metrics_df = evaluate_scores(latent_splits.y_val, s_val, latent_splits.y_test, s_test, CFG)
    save_table(metrics_df, "latentgmm_metrics")

    parents = _load_available_parent_scores()
    ci_df, paired_df = compare_with_parents(
        latent_splits.y_val,
        s_val,
        latent_splits.y_test,
        s_test,
        parents,
        n_boot=n_boot,
        seed=seed,
    )
    save_table(ci_df, "latentgmm_parent_comparison")
    save_table(paired_df, "latentgmm_paired_bootstrap")

    # Task B Evidence (i)-(iii)
    param_df = compute_parameter_count_comparison(
        ambient_dims=(30, splits.X_train_legit.shape[1]),
        latent_dims=sorted({2, 4, 8, 16, latent_splits.latent_dim}),
    )
    save_table(param_df, "latentgmm_param_counts")

    reg_df = run_reg_covar_sensitivity(
        splits.X_train_legit,
        splits.X_val,
        splits.y_val,
        latent_splits.Z_train_legit,
        latent_splits.Z_val,
        n_components=chosen_cfg["n_components"],
        seed=seed,
    )
    save_table(reg_df, "latentgmm_reg_covar_sensitivity")

    seed_df = run_seed_stability_comparison(
        splits.X_train_legit,
        splits.X_val,
        splits.y_val,
        latent_splits.Z_train_legit,
        latent_splits.Z_val,
        ambient_cfg={"covariance_type": "full", "n_components": 12, "reg_covar": 1e-2},
        latent_cfg=chosen_cfg,
    )
    save_table(seed_df, "latentgmm_seed_stability")

    save_fig(plot_latent_aic_bic(grid_df, default_reg=chosen_cfg["reg_covar"]), "latentgmm_aic_bic")
    save_fig(plot_latent_nll_distribution(latent_splits.y_val, s_val, latent_splits.y_test, s_test), "latentgmm_nll_distribution")
    save_fig(plot_latent_stability_evidence(reg_df, seed_df), "latentgmm_stability_comparison")
    plt.close("all")

    return {
        "chosen_cfg": chosen_cfg,
        "metrics_df": metrics_df,
        "ci_df": ci_df,
        "paired_df": paired_df,
        "latent_splits": latent_splits,
        "s_val": s_val,
        "s_test": s_test,
    }


def run_day3_stage(
    feature_set: str = "base",
    checkpoint_path: Path | None = None,
    ablation_checkpoints: Mapping[int, Path] | None = None,
) -> dict[str, Any]:
    """
    Execute Day 3 H1 Latent-GMM experiments:
    - Task B (iv)-(v): Latent vs. ambient Gaussianity, QQ/2D projection plots,
      honest failure analysis, and alert complementarity vs. parent models.
    - Task C: Multi-dim (2/4/8/12/16) x cov-type x K ablation across 3 seeds.
    """
    seed = int(CFG.get("seed", 42))
    set_seed(seed)

    encoder, encoder_meta = load_vani_encoder(checkpoint_path=checkpoint_path)
    splits = get_features(feature_set=feature_set)
    latent_splits = encode_splits(
        encoder,
        splits,
        feature_set=feature_set,
        encoder_id=Path(encoder_meta["checkpoint_path"]).stem,
    )

    s_val = load_scores("hybrid_latentgmm", "val")
    s_test = load_scores("hybrid_latentgmm", "test")
    parents = _load_available_parent_scores()

    gauss_df = analyze_latent_gaussianity(
        splits.X_train_legit,
        latent_splits.Z_train_legit,
        feature_names=splits.feature_names,
        seed=seed,
    )
    save_table(gauss_df, "latentgmm_gaussianity")
    save_table(summarize_gaussianity(gauss_df), "latentgmm_gaussianity_summary")

    val_parent_scores = {k: v["val"] for k, v in parents.items()}
    test_parent_scores = {k: v["test"] for k, v in parents.items()}
    fail_df = analyze_latentgmm_failures(
        latent_splits.y_val,
        s_val,
        parent_scores=val_parent_scores,
        Z_split=latent_splits.Z_val,
    )
    save_table(fail_df, "latentgmm_failure_analysis")

    comp_val = compute_complementarity_summary(latent_splits.y_val, s_val, val_parent_scores)
    comp_val["split"] = "val"
    comp_test = compute_complementarity_summary(latent_splits.y_test, s_test, test_parent_scores)
    comp_test["split"] = "test"
    save_table(pd.concat([comp_val, comp_test], ignore_index=True), "latentgmm_complementarity")

    save_fig(plot_latent_qq(latent_splits.Z_train_legit), "latentgmm_qq_plots")
    save_fig(plot_latent_2d_projection(latent_splits.Z_val, latent_splits.y_val, seed=seed), "latentgmm_2d_projection")
    plt.close("all")

    ablation_map: dict[int, LatentSplits] = {latent_splits.latent_dim: latent_splits}
    ckpt_map = dict(ablation_checkpoints) if ablation_checkpoints is not None else discover_ablation_checkpoints()
    for d_lat, ckpt in ckpt_map.items():
        if d_lat in ablation_map:
            continue
        enc_d, _ = load_vani_encoder(checkpoint_path=ckpt)
        ablation_map[d_lat] = encode_splits(
            enc_d,
            splits,
            feature_set=feature_set,
            encoder_id=ckpt.stem,
        )

    runs_df, summary_df = run_gmm_ablation(ablation_map)
    save_table(runs_df, "ablation_gmm_runs")
    save_table(summary_df, "ablation_gmm_summary")

    return {
        "gauss_summary": summarize_gaussianity(gauss_df),
        "ablation_summary": summary_df,
    }


def run_latent_gmm_stage(
    feature_set: str = "base",
    checkpoint_path: Path | None = None,
    ablation_checkpoints: Mapping[int, Path] | None = None,
    selection_criterion: str = "val_pr_auc",
    n_boot: int = 1000,
) -> dict[str, Any]:
    """
    Execute the complete H1 Latent-GMM pipeline (Day 2 + Day 3 stages).

    Fails immediately with :class:`WaitingForInputsError` if Vani's
    ``src/autoencoder.py`` or ``results/models/ae_*.pt`` weights are absent.
    """
    day2 = run_day2_stage(
        feature_set=feature_set,
        checkpoint_path=checkpoint_path,
        selection_criterion=selection_criterion,
        n_boot=n_boot,
    )
    day3 = run_day3_stage(
        feature_set=feature_set,
        checkpoint_path=checkpoint_path,
        ablation_checkpoints=ablation_checkpoints,
    )
    return {
        "chosen_cfg": day2["chosen_cfg"],
        "metrics_df": day2["metrics_df"],
        "ci_df": day2["ci_df"],
        "paired_df": day2["paired_df"],
        "ablation_summary": day3["ablation_summary"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 H1 Latent-GMM pipeline (Apoorva)")
    parser.add_argument(
        "--feature-set",
        default="base",
        help="Tabular feature set to encode (default: base).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit path to Vani's trained AE checkpoint (.pt).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "day2", "day3"),
        default="all",
        help="Which experimental stage to execute (default: all).",
    )
    parser.add_argument(
        "--stub-test",
        action="store_true",
        help="Run in-memory synthetic stub verification (writes nothing to results/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.stub_test:
        run_synthetic_stub_self_test()
    else:
        try:
            if args.stage == "day2":
                run_day2_stage(feature_set=args.feature_set, checkpoint_path=args.checkpoint)
            elif args.stage == "day3":
                run_day3_stage(feature_set=args.feature_set, checkpoint_path=args.checkpoint)
            else:
                run_latent_gmm_stage(feature_set=args.feature_set, checkpoint_path=args.checkpoint)
        except WaitingForInputsError as exc:
            _logger.error("%s", exc)
            sys.exit(2)

