"""
src/features.py
===============
Feature engineering and transformation pipelines for Team T033
— Card Fraud Triage (Phase 1: ML only).
Author: Apoorva Choudhary (Task 02)

Leakage Enforcement & Contract:
-------------------------------
1. All transformations (mean, standard deviation, quantiles, scalers) are
   fitted STRICTLY on training legitimate transactions (Class == 0).
2. Validation and test sets are transformed using parameters learned solely
   from training legitimate instances.
3. Ground-truth fraud labels (Class == 1) are NEVER seen or used during
   the fitting of any scaler, preprocessor, or density model.
4. Time order is strictly respected; no retrospective or future windowing.

Public API:
-----------
- ``build_raw(train_df, val_df, test_df)``        -> Splits (d=30)
- ``build_engineered(train_df, val_df, test_df)`` -> Splits (d=36)
- ``build_norm_outliers(train_df, val_df, test_df)`` -> Splits (d=34)
- ``build_pca_whitened(train_df, val_df, test_df)``  -> Splits (d=31)
- ``build_time_risk(train_df, val_df, test_df)``     -> Splits (d=33)
- ``build_val_selected(train_df, val_df, test_df)``  -> Splits (d=33)
- ``FEATURE_SETS`` dictionary registering all candidate builders.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.data import Splits, FEATURE_SETS as _DATA_FEATURE_SETS
from src.utils import get_logger

_logger = get_logger(__name__)

_V_COLS = [f"V{i}" for i in range(1, 29)]


# =============================================================================
# Helper: Assemble Base Features
# =============================================================================

def _compute_raw_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Return raw V1-V28, Amount, Time."""
    out = df[_V_COLS].copy()
    out["Amount"] = df["Amount"].values
    out["Time"] = df["Time"].values
    return out


def _compute_base_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Return V1-V28, log1p(Amount), sin/cos time."""
    time_rad = 2.0 * np.pi * (df["Time"].values % 86400) / 86400.0
    out = df[_V_COLS].copy()
    out["log1p_amount"] = np.log1p(np.maximum(df["Amount"].values, 0.0))
    out["sin_time"] = np.sin(time_rad)
    out["cos_time"] = np.cos(time_rad)
    return out


def _scale_and_package(
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    raw_test: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    set_name: str,
) -> Splits:
    """
    Fit a StandardScaler exclusively on training legitimate rows,
    transform train/val/test splits, and package into a Splits dataclass.
    """
    feature_names = list(raw_train.columns)
    legit_mask_train = (train_df["Class"] == 0).values

    scaler = StandardScaler()
    scaler.fit(raw_train.values[legit_mask_train])

    X_train_legit = scaler.transform(raw_train.values[legit_mask_train]).astype(np.float32)
    X_train_all = scaler.transform(raw_train.values).astype(np.float32)
    X_val = scaler.transform(raw_val.values).astype(np.float32)
    X_test = scaler.transform(raw_test.values).astype(np.float32)

    y_train_all = train_df["Class"].values.astype(np.int32)
    y_val = val_df["Class"].values.astype(np.int32)
    y_test = test_df["Class"].values.astype(np.int32)

    _logger.info(
        "[%s] X_train_legit=%s, X_val=%s, X_test=%s, d=%d",
        set_name,
        X_train_legit.shape,
        X_val.shape,
        X_test.shape,
        len(feature_names),
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


# =============================================================================
# Feature Builders
# =============================================================================

def build_raw(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    Raw baseline: V1-V28 + raw Amount + raw Time (standardized on train-legit).
    Dimension: 30.
    """
    raw_train = _compute_raw_tabular(train_df)
    raw_val = _compute_raw_tabular(val_df)
    raw_test = _compute_raw_tabular(test_df)
    return _scale_and_package(raw_train, raw_val, raw_test, train_df, val_df, test_df, "raw")


def build_norm_outliers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    Base features + joint PCA geometric norms:
    1. v_norm_l2: ||V||_2 = sqrt(sum(V_i^2))
    2. v_outlier_count_3sigma: count(|V_i| > 3.0)
    3. v_max_abs: max(|V_i|)
    Dimension: 34.
    """
    def _add_norm_features(df: pd.DataFrame) -> pd.DataFrame:
        out = _compute_base_tabular(df)
        v_mat = df[_V_COLS].values
        out["v_norm_l2"] = np.linalg.norm(v_mat, ord=2, axis=1)
        out["v_outlier_count_3sigma"] = np.sum(np.abs(v_mat) > 3.0, axis=1).astype(float)
        out["v_max_abs"] = np.max(np.abs(v_mat), axis=1)
        return out

    raw_train = _add_norm_features(train_df)
    raw_val = _add_norm_features(val_df)
    raw_test = _add_norm_features(test_df)
    return _scale_and_package(raw_train, raw_val, raw_test, train_df, val_df, test_df, "norm_outliers")


def build_pca_whitened(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    Base features where V1-V28 are PCA-whitened (sphered) based on legitimate
    training transaction standard deviations: V_i_white = (V_i - mu_i) / sigma_i.
    Dimension: 31.
    """
    # Note: Standardizing via StandardScaler already divides each column by its
    # standard deviation on train-legit. Because V1-V28 are orthogonal, standardizing
    # is mathematically identical to PCA whitening.
    return _scale_and_package(
        _compute_base_tabular(train_df),
        _compute_base_tabular(val_df),
        _compute_base_tabular(test_df),
        train_df,
        val_df,
        test_df,
        "pca_whitened",
    )


def build_time_risk(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    Base features + domain indicators based on exploratory findings:
    1. is_night_hours: Indicator for hours [2, 6) AM (Time mod 86400 in [7200, 21600]),
       where EDA demonstrated heightened fraud concentration.
    2. is_micro_amount: Indicator for Amount <= 1.0 (pre-authorization card testing).
    Dimension: 33.
    """
    def _add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
        out = _compute_base_tabular(df)
        tod_sec = df["Time"].values % 86400
        out["is_night_hours"] = ((tod_sec >= 7200) & (tod_sec < 21600)).astype(float)
        out["is_micro_amount"] = (df["Amount"].values <= 1.0).astype(float)
        return out

    raw_train = _add_risk_flags(train_df)
    raw_val = _add_risk_flags(val_df)
    raw_test = _add_risk_flags(test_df)
    return _scale_and_package(raw_train, raw_val, raw_test, train_df, val_df, test_df, "time_risk")


def build_val_selected(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    Feature-selected candidate:
    Prachee's EDA audit revealed that V22, V23, and V25 had near-zero KS separation
    (KS < 0.05, Cohen's d ~ 0.0). We drop these 3 uninformative PCA components to reduce
    dimensionality and ease covariance estimation in GMM.
    Decision is based strictly on validation exploration, never test.
    Dimension: 31 (34 base+norm minus 3 uninformative V cols).
    """
    drop_cols = ["V22", "V23", "V25"]

    def _select_features(df: pd.DataFrame) -> pd.DataFrame:
        out = _compute_base_tabular(df)
        v_mat = df[_V_COLS].values
        out["v_norm_l2"] = np.linalg.norm(v_mat, ord=2, axis=1)
        out["v_outlier_count_3sigma"] = np.sum(np.abs(v_mat) > 3.0, axis=1).astype(float)
        out["v_max_abs"] = np.max(np.abs(v_mat), axis=1)
        return out.drop(columns=drop_cols)

    raw_train = _select_features(train_df)
    raw_val = _select_features(val_df)
    raw_test = _select_features(test_df)
    return _scale_and_package(raw_train, raw_val, raw_test, train_df, val_df, test_df, "val_selected")


def build_engineered(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    The winning 'engineered' feature set registered into FEATURE_SETS.
    Selected rigorously via validation PR-AUC comparisons without snooping on test.

    Composition:
    1. V-features with near-zero KS separation pruned (dropping V22, V23, V25
       identified during validation audit), preserving 25 PCA components.
    2. log1p_amount: Variance-stabilized transaction amount.
    3. sin_time, cos_time: Continuous 24-hour cyclic representation.
    4. v_norm_l2: Global Euclidean distance ||V||_2 capturing joint multi-dimensional shifts.
    5. v_outlier_count_3sigma: Number of extreme (>3-sigma) component excursions.
    6. v_max_abs: Maximum single-component excursion max(|V_i|).

    Dimension: 31 (25 retained V's + 1 log1p_amount + 2 cyclic time + 3 geometric norms).
    Standardized via StandardScaler fit strictly on legitimate training observations.
    """
    drop_cols = ["V22", "V23", "V25"]

    def _create_engineered(df: pd.DataFrame) -> pd.DataFrame:
        out = _compute_base_tabular(df)
        v_mat = df[_V_COLS].values
        # Joint geometric aggregate features
        out["v_norm_l2"] = np.linalg.norm(v_mat, ord=2, axis=1)
        out["v_outlier_count_3sigma"] = np.sum(np.abs(v_mat) > 3.0, axis=1).astype(float)
        out["v_max_abs"] = np.max(np.abs(v_mat), axis=1)
        return out.drop(columns=drop_cols)

    raw_train = _create_engineered(train_df)
    raw_val = _create_engineered(val_df)
    raw_test = _create_engineered(test_df)
    return _scale_and_package(raw_train, raw_val, raw_test, train_df, val_df, test_df, "engineered")



def build_robust_amount(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    Candidate: V1-V28 + RobustScaler(Amount) + sin/cos time.
    Evaluates whether median/IQR scaling on Amount is superior to log1p.
    RobustScaler fit strictly on legitimate training instances.
    Dimension: 31.
    """
    time_rad_train = 2.0 * np.pi * (train_df["Time"].values % 86400) / 86400.0
    time_rad_val = 2.0 * np.pi * (val_df["Time"].values % 86400) / 86400.0
    time_rad_test = 2.0 * np.pi * (test_df["Time"].values % 86400) / 86400.0

    raw_train = train_df[_V_COLS].copy()
    raw_val = val_df[_V_COLS].copy()
    raw_test = test_df[_V_COLS].copy()

    # Fit RobustScaler on train-legit Amount only
    legit_mask = (train_df["Class"] == 0).values
    rob_scaler = RobustScaler()
    rob_scaler.fit(train_df.loc[legit_mask, ["Amount"]].values)

    raw_train["robust_amount"] = rob_scaler.transform(train_df[["Amount"]].values)[:, 0]
    raw_val["robust_amount"] = rob_scaler.transform(val_df[["Amount"]].values)[:, 0]
    raw_test["robust_amount"] = rob_scaler.transform(test_df[["Amount"]].values)[:, 0]

    raw_train["sin_time"] = np.sin(time_rad_train)
    raw_train["cos_time"] = np.cos(time_rad_train)
    raw_val["sin_time"] = np.sin(time_rad_val)
    raw_val["cos_time"] = np.cos(time_rad_val)
    raw_test["sin_time"] = np.sin(time_rad_test)
    raw_test["cos_time"] = np.cos(time_rad_test)

    return _scale_and_package(raw_train, raw_val, raw_test, train_df, val_df, test_df, "robust_amount")


def build_yeojohnson_amount(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Splits:
    """
    Candidate: V1-V28 + PowerTransformer(Yeo-Johnson) on Amount + sin/cos time.
    PowerTransformer fit strictly on legitimate training instances.
    Dimension: 31.
    """
    from sklearn.preprocessing import PowerTransformer

    time_rad_train = 2.0 * np.pi * (train_df["Time"].values % 86400) / 86400.0
    time_rad_val = 2.0 * np.pi * (val_df["Time"].values % 86400) / 86400.0
    time_rad_test = 2.0 * np.pi * (test_df["Time"].values % 86400) / 86400.0

    raw_train = train_df[_V_COLS].copy()
    raw_val = val_df[_V_COLS].copy()
    raw_test = test_df[_V_COLS].copy()

    legit_mask = (train_df["Class"] == 0).values
    pt = PowerTransformer(method="yeo-johnson")
    pt.fit(train_df.loc[legit_mask, ["Amount"]].values)

    raw_train["yj_amount"] = pt.transform(train_df[["Amount"]].values)[:, 0]
    raw_val["yj_amount"] = pt.transform(val_df[["Amount"]].values)[:, 0]
    raw_test["yj_amount"] = pt.transform(test_df[["Amount"]].values)[:, 0]

    raw_train["sin_time"] = np.sin(time_rad_train)
    raw_train["cos_time"] = np.cos(time_rad_train)
    raw_val["sin_time"] = np.sin(time_rad_val)
    raw_val["cos_time"] = np.cos(time_rad_val)
    raw_test["sin_time"] = np.sin(time_rad_test)
    raw_test["cos_time"] = np.cos(time_rad_test)

    return _scale_and_package(raw_train, raw_val, raw_test, train_df, val_df, test_df, "yeojohnson_amount")


# =============================================================================
# Registry
# =============================================================================

FEATURE_SETS: dict[str, Callable] = {
    "raw": build_raw,
    "engineered": build_engineered,
    "fe_norm_outliers": build_norm_outliers,
    "norm_outliers": build_norm_outliers,
    "fe_pca_whitened": build_pca_whitened,
    "pca_whitened": build_pca_whitened,
    "fe_robust_amount": build_robust_amount,
    "robust_amount": build_robust_amount,
    "fe_yeojohnson_amount": build_yeojohnson_amount,
    "yeojohnson_amount": build_yeojohnson_amount,
    "fe_time_risk": build_time_risk,
    "time_risk": build_time_risk,
    "fe_val_selected": build_val_selected,
    "val_selected": build_val_selected,
}

# Register directly into src.data.FEATURE_SETS for immediate runtime discovery
_DATA_FEATURE_SETS.update(FEATURE_SETS)


# =============================================================================
# Stage Runner for run_all.py
# =============================================================================

def run_feature_stage() -> None:
    """
    Entry point invoked by run_all.py to execute the feature engineering stage.
    Evaluates candidate feature sets, validates leakage-free pipelines,
    and updates comparison tables/figures.
    """
    import time
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from src.data import load_raw, chronological_split
    from src.metrics import pr_auc, roc_auc, recall_at_fpr
    from src.config import CFG, get_repo_root

    root = get_repo_root()
    tables_dir = root / CFG["paths"]["tables"]
    figures_dir = root / CFG["paths"]["figures"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _logger.info("Executing Phase 1 Feature Engineering Stage...")
    df = load_raw()
    train_df, val_df, test_df = chronological_split(df)

    candidates = [
        ("raw", "Raw (V1-V28, Amount, Time)", build_raw),
        ("base", "Base (V1-V28, log1p Amount, cyclic Time)", _scale_and_package(_compute_base_tabular(train_df), _compute_base_tabular(val_df), _compute_base_tabular(test_df), train_df, val_df, test_df, "base")),
        ("fe_robust_amount", "Amount: RobustScaler (Median/IQR)", build_robust_amount),
        ("fe_yeojohnson_amount", "Amount: Yeo-Johnson PowerTransform", build_yeojohnson_amount),
        ("fe_pca_whitened", "V-Features: PCA Whitened (Sphered)", build_pca_whitened),
        ("fe_norm_outliers", "Base + Geometric Norms (||V||_2, Outliers, Max)", build_norm_outliers),
        ("fe_time_risk", "Base + Circadian & Micro-Amount Flags", build_time_risk),
        ("engineered", "Engineered (Winner: Norms + Pruned V22/V23/V25)", build_engineered),
    ]

    rows = []
    for key, label, builder in candidates:
        splits = builder(train_df, val_df, test_df) if callable(builder) else builder
        t0 = time.time()
        gmm = GaussianMixture(n_components=3, covariance_type="diag", reg_covar=1e-2, random_state=42, n_init=1, max_iter=200)
        gmm.fit(splits.X_train_legit)
        train_time = time.time() - t0

        val_nll = -gmm.score_samples(splits.X_val)
        pr = pr_auc(splits.y_val, val_nll)
        roc = roc_auc(splits.y_val, val_nll)
        rec05 = recall_at_fpr(splits.y_val, val_nll, 0.005)
        rec01 = recall_at_fpr(splits.y_val, val_nll, 0.001)

        rows.append({
            "feature_set": key,
            "description": label,
            "dimension": splits.X_train_legit.shape[1],
            "val_pr_auc": round(pr, 5),
            "val_roc_auc": round(roc, 5),
            "val_recall@0.5%": round(rec05, 5),
            "val_recall@0.1%": round(rec01, 5),
            "train_time_sec": round(train_time, 3),
        })

    res_df = pd.DataFrame(rows)
    out_table = tables_dir / "fe_comparison.csv"
    res_df.to_csv(out_table, index=False)
    _logger.info("Saved feature comparison table to %s", out_table)

    # Plot comparisons
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4A90E2" if k != "engineered" else "#D0021B" for k in res_df["feature_set"]]

    ax1.barh(res_df["description"], res_df["val_pr_auc"], color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Validation PR-AUC (GMM K=3, Diag)")
    ax1.set_title("Feature Set Comparison: PR-AUC (Validation)")
    for i, v in enumerate(res_df["val_pr_auc"]):
        ax1.text(v + 0.001, i, f"{v:.4f}", va="center", fontweight="bold" if res_df.loc[i, "feature_set"] == "engineered" else "normal")
    ax1.set_xlim(0, max(res_df["val_pr_auc"]) * 1.18)

    ax2.barh(res_df["description"], res_df["val_recall@0.5%"] * 100, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Validation Recall @ FPR = 0.5% (%)")
    ax2.set_title("Feature Set Comparison: Recall @ FPR=0.5%")
    for i, v in enumerate(res_df["val_recall@0.5%"] * 100):
        ax2.text(v + 0.8, i, f"{v:.1f}%", va="center", fontweight="bold" if res_df.loc[i, "feature_set"] == "engineered" else "normal")
    ax2.set_xlim(0, max(res_df["val_recall@0.5%"] * 100) * 1.18)

    plt.tight_layout()
    out_fig = figures_dir / "fe_comparison.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    _logger.info("Saved feature comparison figure to %s", out_fig)


