"""
src/data.py
===========
Data loading, chronological splitting, and feature extraction for Team T033
— Card Fraud Triage (Phase 1: ML only).

Public API
----------
- ``load_raw()``              – read creditcard.csv, validate schema & dtypes.
- ``chronological_split(df)`` – time-ordered 60/20/20 split, saves CSVs + manifest.
- ``get_features(feature_set)`` → ``Splits`` dataclass with all arrays + scaler.

Feature-set registry
---------------------
``FEATURE_SETS`` maps string names to builder functions.  Currently only
``"base"`` is registered here.  Apoorva registers ``"engineered"`` via
``src/features.py`` (imported with try/except so Phase 1 runs without it).

Leakage rules (enforced throughout)
-------------------------------------
1. Scaler is fit on **training legitimate rows only** (Class == 0).
2. Fraud labels are NEVER passed to any anomaly model.
3. ``X_train_all`` exists ONLY for diagnostics; never fit models on it.
4. Seed is loaded from ``CFG['seed']`` (42).

ASSUMPTION
----------
``Time`` represents seconds elapsed from an unknown reference point.  We
assume Time=0 corresponds approximately to midnight of day 1 (i.e. modulo
86 400 s gives a rough hour-of-day proxy).  This assumption is documented here
because the dataset description does not confirm it.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import CFG, get_repo_root
from src.utils import get_logger, set_seed

_logger = get_logger(__name__)
_ROOT: Path = get_repo_root()

# ── Expected schema ──────────────────────────────────────────────────────────
_REQUIRED_COLS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
_N_COLS_EXPECTED = 31  # Time + V1..V28 + Amount + Class


# ═══════════════════════════════════════════════════════════════════════════
# 1. Raw data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_raw() -> pd.DataFrame:
    """
    Read the raw Credit Card Fraud dataset from disk, validate its schema, and
    report duplicate rows.

    Returns
    -------
    pd.DataFrame
        The raw dataset with 284,807 rows and 31 columns
        (Time, V1–V28, Amount, Class) when no rows are dropped.

    Raises
    ------
    FileNotFoundError
        If ``creditcard.csv`` is not found at ``data/raw/creditcard.csv``.
    ValueError
        If the schema (column names, dtypes, expected shape) does not match.

    Notes
    -----
    - Duplicate rows: we *log* the count but do **not** silently drop them.
      Set ``drop_duplicates: true`` in ``configs/config.yaml`` to drop them.
      Default is ``false`` (keep duplicates).
    - This function is intentionally verbose so students can trace every step
      during a viva.
    """
    raw_path = _ROOT / CFG["paths"]["raw"]

    # ── 1a. Check file exists ──────────────────────────────────────────────
    if not raw_path.exists():
        raise FileNotFoundError(
            f"\n[data.py] Cannot find dataset at: {raw_path}\n"
            "Download 'creditcard.csv' from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it at:  data/raw/creditcard.csv"
        )

    _logger.info("Loading raw data from %s …", raw_path)
    df = pd.read_csv(raw_path)
    _logger.info("Loaded: %d rows × %d cols", *df.shape)

    # ── 1b. Schema validation ──────────────────────────────────────────────
    missing_cols = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[data.py] Missing expected columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    if df.shape[1] != _N_COLS_EXPECTED:
        raise ValueError(
            f"[data.py] Expected {_N_COLS_EXPECTED} columns, got {df.shape[1]}."
        )

    # ── 1c. Dtype checks ───────────────────────────────────────────────────
    for col in _REQUIRED_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"[data.py] Column '{col}' is not numeric (dtype={df[col].dtype})."
            )
    if set(df["Class"].unique()) - {0, 1}:
        raise ValueError(
            f"[data.py] 'Class' column contains values other than 0/1: "
            f"{sorted(df['Class'].unique())}"
        )

    # ── 1d. Duplicate rows ─────────────────────────────────────────────────
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        _logger.warning(
            "Found %d exact duplicate rows (%.4f%% of dataset).",
            n_dups, 100 * n_dups / len(df),
        )
        if CFG.get("drop_duplicates", False):
            df = df.drop_duplicates().reset_index(drop=True)
            _logger.info("Dropped duplicates. Remaining rows: %d", len(df))
        else:
            _logger.info(
                "Keeping duplicates (drop_duplicates=false in config). "
                "Set drop_duplicates: true to remove them."
            )
    else:
        _logger.info("No exact duplicate rows found.")

    # ── 1e. Basic sanity stats ─────────────────────────────────────────────
    n_fraud = int((df["Class"] == 1).sum())
    _logger.info(
        "Frauds: %d / %d  (base rate = %.4f%%)",
        n_fraud, len(df), 100 * n_fraud / len(df),
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Chronological split
# ═══════════════════════════════════════════════════════════════════════════

def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically (by ``Time`` column) into train, val,
    and test partitions.

    The split fractions come from ``CFG['split']`` (default 60/20/20).
    NO SHUFFLING is applied; rows are simply cut at the appropriate index after
    sorting by ``Time`` with a stable sort.

    Saves to disk:
    - ``data/processed/train.csv``, ``val.csv``, ``test.csv``
      (original columns + original index preserved)
    - ``data/processed/split_manifest.json``
      (row counts, fraud counts, fraud rate, Time min/max per split)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by ``load_raw()``.

    Returns
    -------
    tuple of (train_df, val_df, test_df) — all as DataFrames.

    Raises
    ------
    ValueError
        If the fractions in config do not sum to 1.0 (within tolerance).

    Warnings
    --------
    Prints a loud warning if val or test contains fewer than 50 fraud rows
    and suggests using a 60/15/25 split instead.
    """
    train_frac = float(CFG["split"]["train_frac"])
    val_frac   = float(CFG["split"]["val_frac"])
    test_frac  = float(CFG["split"]["test_frac"])

    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError(
            f"Split fractions must sum to 1.0, got "
            f"{train_frac} + {val_frac} + {test_frac} = "
            f"{train_frac + val_frac + test_frac}"
        )

    # ── Sort by Time (stable, preserves original ordering for equal times) ─
    df_sorted = df.sort_values("Time", kind="stable").reset_index(drop=False)
    # 'index' column now holds the original DataFrame index.

    n = len(df_sorted)
    n_train = int(np.floor(n * train_frac))
    n_val   = int(np.floor(n * val_frac))
    # test gets the remainder to avoid rounding loss
    n_test  = n - n_train - n_val

    train_df = df_sorted.iloc[:n_train].copy()
    val_df   = df_sorted.iloc[n_train : n_train + n_val].copy()
    test_df  = df_sorted.iloc[n_train + n_val :].copy()

    assert len(train_df) + len(val_df) + len(test_df) == n, "Split sizes don't add up."

    # ── Warn if val/test have < 50 frauds ─────────────────────────────────
    _MIN_FRAUD = 50
    for name, split in [("val", val_df), ("test", test_df)]:
        n_fraud_split = int((split["Class"] == 1).sum())
        if n_fraud_split < _MIN_FRAUD:
            warnings.warn(
                f"\n{'='*60}\n"
                f"WARNING: '{name}' split has only {n_fraud_split} fraud rows "
                f"(< {_MIN_FRAUD}).\n"
                f"Threshold estimates and PR-AUC confidence intervals will be\n"
                f"unstable.  Consider changing to 60/15/25 split in config.\n"
                f"Document the final decision in docs/data_card.md.\n"
                f"{'='*60}",
                stacklevel=2,
            )

    # ── Save split CSVs ───────────────────────────────────────────────────
    processed_dir = _ROOT / CFG["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Restore the original index (drop the temporary 'index' column added by reset_index)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = processed_dir / f"{name}.csv"
        # Write with the original index as the CSV index
        split.set_index("index").to_csv(out_path)
        _logger.info(
            "Saved %s.csv  (%d rows, %d frauds, base_rate=%.4f%%)",
            name, len(split),
            int((split["Class"] == 1).sum()),
            100 * (split["Class"] == 1).mean(),
        )

    # ── Save split manifest ───────────────────────────────────────────────
    manifest = {}
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_fraud_split = int((split["Class"] == 1).sum())
        manifest[name] = {
            "n_rows":     int(len(split)),
            "n_fraud":    n_fraud_split,
            "fraud_rate": round(float(n_fraud_split / len(split)), 8),
            "time_min":   float(split["Time"].min()),
            "time_max":   float(split["Time"].max()),
        }

    manifest_path = processed_dir / "split_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    _logger.info("Saved split_manifest.json → %s", manifest_path)

    # Log the time ordering guarantee
    _logger.info(
        "Time ordering check: train_max=%.1f  val_min=%.1f  val_max=%.1f  test_min=%.1f",
        train_df["Time"].max(),
        val_df["Time"].min(),
        val_df["Time"].max(),
        test_df["Time"].min(),
    )

    # Restore DataFrames with original index for return
    return (
        train_df.set_index("index"),
        val_df.set_index("index"),
        test_df.set_index("index"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Feature builders
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class Splits:
    """
    Container for all feature arrays and metadata returned by ``get_features``.

    Attributes
    ----------
    X_train_legit : np.ndarray, shape (n_train_legit, d)
        Scaled features for **legitimate training rows only**.
        Use this to fit anomaly models.
    X_train_all : np.ndarray, shape (n_train, d)
        Scaled features for ALL training rows.
        **For diagnostics only — never fit a model on this.**
    y_train_all : np.ndarray, shape (n_train,)
        Labels for all training rows (used for diagnostics only).
    X_val : np.ndarray, shape (n_val, d)
        Scaled features for the entire validation split.
    y_val : np.ndarray, shape (n_val,)
        Labels for the validation split.
    X_test : np.ndarray, shape (n_test, d)
        Scaled features for the entire test split.
    y_test : np.ndarray, shape (n_test,)
        Labels for the test split.
    feature_names : list[str]
        Names of the *d* features, in column order.
    scaler : StandardScaler
        The scaler fit on ``X_train_legit`` (before scaling).
    """
    X_train_legit: np.ndarray
    X_train_all:   np.ndarray
    y_train_all:   np.ndarray
    X_val:         np.ndarray
    y_val:         np.ndarray
    X_test:        np.ndarray
    y_test:        np.ndarray
    feature_names: list[str]
    scaler:        StandardScaler


def _build_base(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> Splits:
    """
    Build the **base** feature set.

    Features:
    1. ``V1`` … ``V28``          — raw PCA components (standardised).
    2. ``log1p(Amount)``         — log-transformed transaction amount (standardised).
    3. ``sin(2π · (Time mod 86400) / 86400)`` — cyclic time-of-day encoding.
    4. ``cos(2π · (Time mod 86400) / 86400)`` — cyclic time-of-day encoding.

    Raw ``Time`` is DROPPED after encoding.  ``Amount`` is DROPPED after
    log-transforming.

    Scaler: ``StandardScaler`` is fit on **training legitimate rows only**
    (rows where ``Class == 0``), then applied to all splits (train, val, test).

    ASSUMPTION
    ----------
    We assume ``Time`` (seconds elapsed) modulo 86 400 gives a meaningful
    proxy for the hour of the day.  The dataset documentation does not confirm
    that Time=0 corresponds to midnight; this is an approximation.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames as returned by ``chronological_split``.

    Returns
    -------
    Splits
    """
    v_cols  = [f"V{i}" for i in range(1, 29)]

    def _raw_features(df: pd.DataFrame) -> pd.DataFrame:
        """Assemble raw (unscaled) feature matrix from one split."""
        time_angle = 2 * np.pi * (df["Time"].values % 86400) / 86400
        out = df[v_cols].copy()
        out["log1p_amount"] = np.log1p(df["Amount"].values)
        out["sin_time"]     = np.sin(time_angle)
        out["cos_time"]     = np.cos(time_angle)
        return out

    raw_train = _raw_features(train_df)
    raw_val   = _raw_features(val_df)
    raw_test  = _raw_features(test_df)

    feature_names = list(raw_train.columns)

    # ── Fit scaler on training-legit only ────────────────────────────────
    legit_mask_train = (train_df["Class"] == 0).values
    scaler = StandardScaler()
    scaler.fit(raw_train.values[legit_mask_train])

    X_train_legit = scaler.transform(raw_train.values[legit_mask_train]).astype(np.float32)
    X_train_all   = scaler.transform(raw_train.values).astype(np.float32)
    X_val         = scaler.transform(raw_val.values).astype(np.float32)
    X_test        = scaler.transform(raw_test.values).astype(np.float32)

    y_train_all = train_df["Class"].values.astype(np.int32)
    y_val       = val_df["Class"].values.astype(np.int32)
    y_test      = test_df["Class"].values.astype(np.int32)

    _logger.info(
        "[base] X_train_legit=%s  X_val=%s  X_test=%s  features=%d",
        X_train_legit.shape, X_val.shape, X_test.shape, len(feature_names),
    )

    return Splits(
        X_train_legit=X_train_legit,
        X_train_all=X_train_all,
        y_train_all=y_train_all,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )


# ── Feature-set registry ──────────────────────────────────────────────────
# Other members register additional feature sets through src/features.py.
# Do NOT import src.features here unconditionally — it may not exist yet.
FEATURE_SETS: dict[str, Callable] = {
    "base": _build_base,
}

# Attempt to load Apoorva's engineered feature set (Phase 1, non-blocking).
try:
    from src import features as _feat_module  # noqa: F401
    if hasattr(_feat_module, "FEATURE_SETS"):
        FEATURE_SETS.update(_feat_module.FEATURE_SETS)
        _logger.info(
            "Registered additional feature sets from src.features: %s",
            list(_feat_module.FEATURE_SETS.keys()),
        )
except ImportError:
    pass  # src/features.py not yet created — fine for Phase 1 scaffold


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def get_features(feature_set: str | None = None) -> Splits:
    """
    Load the dataset, split it chronologically, and build the requested
    feature set.

    This is the **single function all team members call** to obtain aligned,
    leak-free arrays for model training and evaluation.

    Parameters
    ----------
    feature_set : str, optional
        Name of the feature set to build.  If ``None``, uses
        ``CFG['feature_set']`` (default ``"base"``).

    Returns
    -------
    Splits
        A dataclass with ``X_train_legit``, ``X_train_all``, ``y_train_all``,
        ``X_val``, ``y_val``, ``X_test``, ``y_test``, ``feature_names``,
        and ``scaler``.

    Raises
    ------
    KeyError
        If *feature_set* is not registered in ``FEATURE_SETS``.

    Example
    -------
    >>> from src.data import get_features
    >>> s = get_features()
    >>> print(s.X_val.shape)
    (56962, 31)
    """
    set_seed()  # reproducibility — reads CFG['seed']

    if feature_set is None:
        feature_set = CFG.get("feature_set", "base")

    if feature_set not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature_set='{feature_set}'. "
            f"Available: {sorted(FEATURE_SETS.keys())}"
        )

    # ── Load raw data ─────────────────────────────────────────────────────
    df = load_raw()

    # ── Split chronologically ────────────────────────────────────────────
    train_df, val_df, test_df = chronological_split(df)

    # ── Build features ────────────────────────────────────────────────────
    builder = FEATURE_SETS[feature_set]
    splits = builder(train_df, val_df, test_df)

    _logger.info(
        "get_features('%s') done. "
        "X_train_legit=%s X_val=%s X_test=%s",
        feature_set,
        splits.X_train_legit.shape,
        splits.X_val.shape,
        splits.X_test.shape,
    )
    return splits
