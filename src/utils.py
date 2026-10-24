"""
src/utils.py
============
Shared utilities for Team T033 — Card Fraud Triage (Phase 1: ML only).

Provides:
- ``set_seed``       – set random seeds for reproducibility.
- ``get_logger``     – return a named, configured Python logger.
- ``save_scores``    – save a 1-D float score array to results/scores/.
- ``load_scores``    – load a previously saved score array.
- ``save_table``     – save a pandas DataFrame to results/tables/ as CSV.
- ``save_fig``       – save a matplotlib Figure to results/figures/.

Score-file contract (every team member follows this):
  - 1-D float32 numpy array.
  - Length == len(split).
  - Row order == order of data/processed/{split}.csv.
  - Higher score == more anomalous (more likely fraud).
  - File path: results/scores/{model_name}_{split}.npy
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.config import CFG, get_repo_root

# ── Repo root ────────────────────────────────────────────────────────────────
_ROOT: Path = get_repo_root()


# ═══════════════════════════════════════════════════════════════════════════
# Seed
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int | None = None) -> int:
    """
    Set Python, NumPy, and (if available) PyTorch random seeds for full
    reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value.  Defaults to ``CFG['seed']`` (42).

    Returns
    -------
    int
        The seed that was actually set.

    Notes
    -----
    PyTorch is NOT required for Phase 1; the import is attempted with a
    ``try/except`` so the function works even without it.
    """
    if seed is None:
        seed = int(CFG.get("seed", 42))

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # noqa: F401
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not required for Phase 1

    return seed


# ═══════════════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════════════

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named Python logger with a consistent format.

    Parameters
    ----------
    name : str
        Logger name, e.g. ``__name__`` of the calling module.
    level : int, optional
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger

    Notes
    -----
    Calling this multiple times with the same *name* returns the same logger
    (Python's logging module is a registry).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# Score I/O
# ═══════════════════════════════════════════════════════════════════════════

def _scores_dir() -> Path:
    """Return the scores output directory, creating it if necessary."""
    d = _ROOT / CFG["paths"]["scores"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_scores(model_name: str, split: str, scores: np.ndarray) -> Path:
    """
    Save a 1-D float score array following the team score-file contract.

    Parameters
    ----------
    model_name : str
        Short identifier for the model, e.g. ``"gmm"``, ``"isoforest"``.
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    scores : np.ndarray
        1-D float array; **higher == more anomalous**.

    Returns
    -------
    Path
        The path to the saved ``.npy`` file.

    Raises
    ------
    ValueError
        If *scores* is not a 1-D array.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(
            f"scores must be 1-D, got shape {scores.shape}."
        )
    out = _scores_dir() / f"{model_name}_{split}.npy"
    np.save(out, scores)
    logger = get_logger(__name__)
    logger.info("Saved scores → %s  (n=%d, min=%.4f, max=%.4f)",
                out, len(scores), scores.min(), scores.max())
    return out


def load_scores(model_name: str, split: str) -> np.ndarray:
    """
    Load a previously saved score array.

    Parameters
    ----------
    model_name : str
        Short model identifier used when saving.
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.

    Returns
    -------
    np.ndarray
        1-D float32 array.

    Raises
    ------
    FileNotFoundError
        If the ``.npy`` file does not exist.
    """
    path = _scores_dir() / f"{model_name}_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Score file not found: {path}\n"
            "Has this model been run yet?"
        )
    return np.load(path).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Table I/O
# ═══════════════════════════════════════════════════════════════════════════

def _tables_dir() -> Path:
    """Return the tables output directory, creating it if necessary."""
    d = _ROOT / CFG["paths"]["tables"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_table(df: pd.DataFrame, name: str) -> Path:
    """
    Save a pandas DataFrame as a CSV file to results/tables/.

    Parameters
    ----------
    df : pd.DataFrame
        The table to save.
    name : str
        Filename stem (without extension), e.g. ``"gmm_metrics"``.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    out = _tables_dir() / f"{name}.csv"
    df.to_csv(out)
    logger = get_logger(__name__)
    logger.info("Saved table → %s  (%d rows × %d cols)", out, *df.shape)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure I/O
# ═══════════════════════════════════════════════════════════════════════════

def _figures_dir() -> Path:
    """Return the figures output directory, creating it if necessary."""
    d = _ROOT / CFG["paths"]["figures"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, name: str, dpi: int = 150) -> Path:
    """
    Save a matplotlib Figure to results/figures/ as a PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    name : str
        Filename stem (without extension), e.g. ``"gmm_pr_curve"``.
    dpi : int, optional
        Resolution in dots per inch (default 150).

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    out = _figures_dir() / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger = get_logger(__name__)
    logger.info("Saved figure → %s", out)
    return out
