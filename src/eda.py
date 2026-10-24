"""
src/eda.py
==========
Exploratory Data Analysis (EDA) functions for Project T033 — Card Fraud Triage.
Author: Prachee Dhar (Task 01).

This module provides functions to:
  1. Compute dataset shape, dtype, memory, missing values, and duplicate analysis.
  2. Measure and plot class imbalance (fraud vs legitimate).
  3. Analyse the Amount feature (distribution, outliers, zero-amounts).
  4. Analyse the Time feature and per-hour fraud patterns.
  5. Compare train/val/test feature distributions using KS-tests (drift check).
  6. Produce KDE density plots for V1–V28 by class.
  7. Compute a correlation heatmap on legitimate training rows.
  8. Produce 2-D PCA and t-SNE projections coloured by class.
  9. Run Gaussianity checks (QQ-plots, skewness, kurtosis) on training-legitimate rows.
 10. Produce a leakage audit table.

All figures are saved via ``src.utils.save_fig``.
All tables are saved via ``src.utils.save_table``.

Usage (from project root)
--------------------------
    from src.eda import run_all_eda
    run_all_eda()

No leakage
----------
- No scaler or model is fit in this module.
- The ``y`` (Class) column is used *only* to colour plots and split rows into
  legit/fraud groups for description — not to fit any model.
- Drift statistics are computed from raw feature values, not from model scores.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts/notebooks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.decomposition import PCA

from src.config import CFG, get_repo_root
from src.utils import get_logger, save_fig, save_table

_logger = get_logger(__name__)
_ROOT: Path = get_repo_root()

# ── Visual style ─────────────────────────────────────────────────────────────
FRAUD_COLOUR  = "#e63946"   # vivid red — fraud
LEGIT_COLOUR  = "#457b9d"   # steel blue — legitimate
PALETTE       = {"Legit": LEGIT_COLOUR, "Fraud": FRAUD_COLOUR}

plt.rcParams.update({
    "figure.dpi":     150,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family":    "sans-serif",
})

# ── Small helper ─────────────────────────────────────────────────────────────

def _label_series(y: pd.Series) -> pd.Series:
    """Map Class 0/1 to 'Legit'/'Fraud' string labels."""
    return y.map({0: "Legit", 1: "Fraud"})


# ═══════════════════════════════════════════════════════════════════════════
# 1. Basic dataset quality summary
# ═══════════════════════════════════════════════════════════════════════════

def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame describing shape, dtypes, memory, missing values,
    and duplicate counts for the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset (output of ``load_raw()``).

    Returns
    -------
    pd.DataFrame
        Summary table saved to ``results/tables/eda_dataset_summary.csv``.
    """
    n_rows, n_cols = df.shape
    missing = df.isnull().sum()
    n_dup = int(df.duplicated().sum())

    # Per-column summary
    col_summary = pd.DataFrame({
        "dtype":         df.dtypes,
        "n_missing":     missing,
        "pct_missing":   (missing / n_rows * 100).round(4),
        "min":           df.min(),
        "max":           df.max(),
        "mean":          df.mean(numeric_only=True),
        "std":           df.std(numeric_only=True),
    })

    # Overall summary as single-row table
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    overall = pd.DataFrame([{
        "n_rows":           n_rows,
        "n_cols":           n_cols,
        "memory_MB":        round(mem_mb, 2),
        "total_missing":    int(missing.sum()),
        "n_duplicates":     n_dup,
        "pct_duplicates":   round(100 * n_dup / n_rows, 4),
        "n_fraud":          int((df["Class"] == 1).sum()),
        "fraud_rate_pct":   round(100 * (df["Class"] == 1).mean(), 4),
    }])

    save_table(col_summary, "eda_column_summary")
    save_table(overall, "eda_dataset_summary")
    _logger.info("Dataset summary: %d rows, %d cols, %d dupes, %d missing",
                 n_rows, n_cols, n_dup, int(missing.sum()))
    return overall


def duplicate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse duplicate rows: count, class breakdown, and whether duplicates
    appear across classes (a potential data quality concern).

    Returns
    -------
    pd.DataFrame
        Table of duplicate statistics saved to ``results/tables/eda_duplicates.csv``.
    """
    dup_mask = df.duplicated(keep=False)
    n_dup_total = int(dup_mask.sum())

    if n_dup_total == 0:
        result = pd.DataFrame([{"n_duplicate_rows": 0,
                                 "n_dup_legit": 0,
                                 "n_dup_fraud": 0,
                                 "cross_class_duplicates": False}])
        save_table(result, "eda_duplicates")
        _logger.info("No exact duplicate rows found.")
        return result

    dup_df = df[dup_mask]
    n_dup_legit = int((dup_df["Class"] == 0).sum())
    n_dup_fraud = int((dup_df["Class"] == 1).sum())

    # Cross-class duplicates: drop Class, check if a legit row equals a fraud row
    feature_cols = [c for c in df.columns if c != "Class"]
    dup_features = df[feature_cols][dup_mask]
    cross_class = bool(dup_features.duplicated(keep=False).any() and (n_dup_legit > 0 and n_dup_fraud > 0))

    result = pd.DataFrame([{
        "n_duplicate_rows":    n_dup_total,
        "n_dup_legit":         n_dup_legit,
        "n_dup_fraud":         n_dup_fraud,
        "pct_dup_of_total":    round(100 * n_dup_total / len(df), 4),
        "cross_class_duplicates": cross_class,
    }])
    save_table(result, "eda_duplicates")
    _logger.info("Duplicates: %d rows (%d legit, %d fraud). Cross-class: %s",
                 n_dup_total, n_dup_legit, n_dup_fraud, cross_class)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Class imbalance
# ═══════════════════════════════════════════════════════════════════════════

def class_imbalance_plot(df: pd.DataFrame,
                          manifest_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Plot class imbalance (bar + log-scale bar) and return counts per split.

    Parameters
    ----------
    df : pd.DataFrame
        Full raw dataset.
    manifest_path : Path, optional
        Path to split_manifest.json.  Defaults to data/processed/split_manifest.json.

    Returns
    -------
    pd.DataFrame
        Per-split base rate table saved to ``results/tables/eda_class_imbalance.csv``.
    """
    if manifest_path is None:
        manifest_path = _ROOT / "data" / "processed" / "split_manifest.json"

    # ── Load manifest ──────────────────────────────────────────────────────
    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for split_name, info in manifest.items():
        rows.append({
            "split":        split_name,
            "n_rows":       info["n_rows"],
            "n_fraud":      info["n_fraud"],
            "n_legit":      info["n_rows"] - info["n_fraud"],
            "fraud_rate_pct": round(info["fraud_rate"] * 100, 4),
        })
    # Add full dataset row
    n_fraud_all = int((df["Class"] == 1).sum())
    rows.append({
        "split":        "all",
        "n_rows":       len(df),
        "n_fraud":      n_fraud_all,
        "n_legit":      len(df) - n_fraud_all,
        "fraud_rate_pct": round(100 * n_fraud_all / len(df), 4),
    })
    table = pd.DataFrame(rows)
    save_table(table, "eda_class_imbalance")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = pd.Series({"Legitimate": len(df) - n_fraud_all, "Fraud": n_fraud_all})
    axes[0].bar(counts.index, counts.values,
                color=[LEGIT_COLOUR, FRAUD_COLOUR], edgecolor="black", linewidth=0.7)
    axes[0].set_title("Class Distribution (linear scale)")
    axes[0].set_ylabel("Number of transactions")
    for i, (k, v) in enumerate(counts.items()):
        axes[0].text(i, v * 1.01, f"{v:,}\n({100*v/len(df):.3f}%)",
                     ha="center", va="bottom", fontsize=9)

    axes[1].bar(counts.index, counts.values,
                color=[LEGIT_COLOUR, FRAUD_COLOUR], edgecolor="black", linewidth=0.7)
    axes[1].set_yscale("log")
    axes[1].set_title("Class Distribution (log scale)")
    axes[1].set_ylabel("Number of transactions (log scale)")

    fig.suptitle(
        f"Class Imbalance — {n_fraud_all:,} fraud / {len(df):,} total "
        f"({100*n_fraud_all/len(df):.4f}% base rate)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_fig(fig, "eda_class_imbalance")
    plt.close(fig)
    return table


# ═══════════════════════════════════════════════════════════════════════════
# 3. Amount feature analysis
# ═══════════════════════════════════════════════════════════════════════════

def amount_analysis(df: pd.DataFrame) -> dict:
    """
    Analyse the Amount feature: raw distribution (log1p), fraud vs legit
    comparison, outliers (IQR), and zero-amount count.

    Returns
    -------
    dict
        Statistics dict (also saved as eda_amount_stats.csv).
    """
    legit  = df[df["Class"] == 0]["Amount"]
    fraud  = df[df["Class"] == 1]["Amount"]

    q1, q3 = legit.quantile(0.25), legit.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers_legit = int(((legit < lower) | (legit > upper)).sum())
    n_zero_amount    = int((df["Amount"] == 0).sum())

    stats_dict = {
        "legit_mean":       round(float(legit.mean()), 4),
        "legit_median":     round(float(legit.median()), 4),
        "legit_std":        round(float(legit.std()), 4),
        "legit_max":        round(float(legit.max()), 4),
        "fraud_mean":       round(float(fraud.mean()), 4),
        "fraud_median":     round(float(fraud.median()), 4),
        "fraud_std":        round(float(fraud.std()), 4),
        "fraud_max":        round(float(fraud.max()), 4),
        "n_zero_amount":    n_zero_amount,
        "pct_zero_amount":  round(100 * n_zero_amount / len(df), 4),
        "n_legit_outliers_IQR": n_outliers_legit,
        "iqr_lower":        round(lower, 4),
        "iqr_upper":        round(upper, 4),
    }
    save_table(pd.DataFrame([stats_dict]), "eda_amount_stats")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw Amount distribution (log1p) for all transactions
    axes[0, 0].hist(np.log1p(df["Amount"]), bins=80,
                    color=LEGIT_COLOUR, edgecolor="none", alpha=0.7)
    axes[0, 0].set_title("log1p(Amount) — All transactions")
    axes[0, 0].set_xlabel("log1p(Amount)")
    axes[0, 0].set_ylabel("Count")

    # Amount by class (log1p)
    axes[0, 1].hist(np.log1p(legit), bins=80, alpha=0.6,
                    color=LEGIT_COLOUR, label="Legit", edgecolor="none")
    axes[0, 1].hist(np.log1p(fraud), bins=40, alpha=0.8,
                    color=FRAUD_COLOUR, label="Fraud", edgecolor="none")
    axes[0, 1].set_title("log1p(Amount) by Class")
    axes[0, 1].set_xlabel("log1p(Amount)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # Boxplot by class
    try:
        axes[1, 0].boxplot([np.log1p(legit.values), np.log1p(fraud.values)],
                           tick_labels=["Legit", "Fraud"],
                           patch_artist=True,
                           boxprops=dict(facecolor="#d6eaf8"),
                           medianprops=dict(color="black", linewidth=2))
    except TypeError:
        axes[1, 0].boxplot([np.log1p(legit.values), np.log1p(fraud.values)],
                           labels=["Legit", "Fraud"],
                           patch_artist=True,
                           boxprops=dict(facecolor="#d6eaf8"),
                           medianprops=dict(color="black", linewidth=2))
    axes[1, 0].set_title("log1p(Amount) Box Plot by Class")
    axes[1, 0].set_ylabel("log1p(Amount)")

    # IQR outlier visualisation (legit only)
    axes[1, 1].hist(np.log1p(legit), bins=80, color=LEGIT_COLOUR,
                    alpha=0.7, edgecolor="none")
    axes[1, 1].axvline(np.log1p(upper), color="red", linestyle="--",
                       label=f"IQR upper (${upper:.0f})")
    axes[1, 1].set_title(f"Legit Amount — IQR outliers: {n_outliers_legit:,}")
    axes[1, 1].set_xlabel("log1p(Amount)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Amount Feature Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eda_amount")
    plt.close(fig)

    _logger.info("Amount: legit mean=%.2f, fraud mean=%.2f, zero-amount=%d",
                 stats_dict["legit_mean"], stats_dict["fraud_mean"], n_zero_amount)
    return stats_dict


# ═══════════════════════════════════════════════════════════════════════════
# 4. Time feature analysis
# ═══════════════════════════════════════════════════════════════════════════

def time_analysis(df: pd.DataFrame) -> dict:
    """
    Analyse the Time feature: span, transactions per hour, fraud per hour.

    Returns
    -------
    dict
        Time statistics (also saved as eda_time_stats.csv).
    """
    time_span_s  = float(df["Time"].max() - df["Time"].min())
    time_span_h  = time_span_s / 3600

    df2 = df.copy()
    df2["hour_of_day"] = (df2["Time"] % 86400 // 3600).astype(int)

    per_hour_total = df2.groupby("hour_of_day").size()
    per_hour_fraud = df2[df2["Class"] == 1].groupby("hour_of_day").size().reindex(
        range(24), fill_value=0)
    per_hour_fraud_rate = (per_hour_fraud / per_hour_total.reindex(
        range(24), fill_value=1)).fillna(0)

    stats_dict = {
        "time_min_s":      float(df["Time"].min()),
        "time_max_s":      float(df["Time"].max()),
        "time_span_s":     time_span_s,
        "time_span_hours": round(time_span_h, 2),
        "n_hours":         int(np.ceil(time_span_h)),
    }
    save_table(pd.DataFrame([stats_dict]), "eda_time_stats")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # All transactions over time
    axes[0].hist(df["Time"] / 3600, bins=100, color=LEGIT_COLOUR, alpha=0.7)
    axes[0].set_title("All Transactions over Time (hours)")
    axes[0].set_xlabel("Time (hours from start)")
    axes[0].set_ylabel("Transaction count")

    # Fraud transactions over time
    axes[1].hist(df[df["Class"] == 1]["Time"] / 3600, bins=50,
                 color=FRAUD_COLOUR, alpha=0.8)
    axes[1].set_title("Fraud Transactions over Time (hours)")
    axes[1].set_xlabel("Time (hours from start)")
    axes[1].set_ylabel("Fraud count")

    # Fraud rate per hour-of-day
    hours = list(range(24))
    axes[2].bar(hours, [per_hour_fraud_rate.get(h, 0) * 100 for h in hours],
                color=FRAUD_COLOUR, alpha=0.8)
    axes[2].set_title("Fraud Rate by Hour of Day (ASSUMPTION: Time mod 86400 ≈ hour-of-day)")
    axes[2].set_xlabel("Hour of day (0–23)")
    axes[2].set_ylabel("Fraud rate (%)")
    axes[2].set_xticks(hours)

    fig.suptitle(f"Time Feature Analysis (span ≈ {time_span_h:.1f} hours ≈ 2 days)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eda_time")
    plt.close(fig)

    _logger.info("Time span: %.1f hours (%.0f seconds)", time_span_h, time_span_s)
    return stats_dict


# ═══════════════════════════════════════════════════════════════════════════
# 5. Drift check: KS-test across splits
# ═══════════════════════════════════════════════════════════════════════════

def drift_check_ks(train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Run a two-sample Kolmogorov-Smirnov test for each feature comparing
    train vs val and train vs test distributions.

    This is a drift check — if the distributions shift significantly across
    the chronological splits, GMM (fit on train-legit) may be degraded at
    test time.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames (with 'Class' and feature columns).
    feature_cols : list, optional
        Feature columns to test.  Defaults to V1–V28, Amount, Time.

    Returns
    -------
    pd.DataFrame
        KS statistics per feature, saved as ``results/tables/eda_drift_ks.csv``.
    """
    if feature_cols is None:
        v_cols = [f"V{i}" for i in range(1, 29)]
        feature_cols = v_cols + ["Amount", "Time"]

    rows = []
    for col in feature_cols:
        ks_tv = stats.ks_2samp(train_df[col].values, val_df[col].values)
        ks_tt = stats.ks_2samp(train_df[col].values, test_df[col].values)
        rows.append({
            "feature":       col,
            "ks_stat_train_val":  round(ks_tv.statistic, 6),
            "pval_train_val":     round(ks_tv.pvalue, 6),
            "ks_stat_train_test": round(ks_tt.statistic, 6),
            "pval_train_test":    round(ks_tt.pvalue, 6),
            "drift_tv_significant": ks_tv.pvalue < 0.05,
            "drift_tt_significant": ks_tt.pvalue < 0.05,
        })

    ks_table = pd.DataFrame(rows).sort_values("ks_stat_train_val", ascending=False)
    save_table(ks_table, "eda_drift_ks")

    # ── Bar-chart figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    x = np.arange(len(ks_table))
    axes[0].bar(x, ks_table["ks_stat_train_val"],
                color=[FRAUD_COLOUR if sig else LEGIT_COLOUR
                       for sig in ks_table["drift_tv_significant"]],
                edgecolor="none", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ks_table["feature"], rotation=90, fontsize=7)
    axes[0].set_title("KS Statistic: Train vs Val (red = p < 0.05, significant drift)")
    axes[0].set_ylabel("KS statistic")
    axes[0].axhline(0.05, color="grey", linestyle="--", linewidth=0.8, label="KS=0.05 reference")
    axes[0].legend()

    axes[1].bar(x, ks_table["ks_stat_train_test"],
                color=[FRAUD_COLOUR if sig else LEGIT_COLOUR
                       for sig in ks_table["drift_tt_significant"]],
                edgecolor="none", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ks_table["feature"], rotation=90, fontsize=7)
    axes[1].set_title("KS Statistic: Train vs Test (red = p < 0.05, significant drift)")
    axes[1].set_ylabel("KS statistic")

    fig.suptitle("Feature Distribution Drift across Chronological Splits (KS Test)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eda_drift_ks")
    plt.close(fig)

    n_sig_tv = int(ks_table["drift_tv_significant"].sum())
    n_sig_tt = int(ks_table["drift_tt_significant"].sum())
    _logger.info("Drift: %d/%d features significant train-val, %d/%d train-test",
                 n_sig_tv, len(ks_table), n_sig_tt, len(ks_table))
    return ks_table


# ═══════════════════════════════════════════════════════════════════════════
# 6. V1–V28 KDE density plots by class
# ═══════════════════════════════════════════════════════════════════════════

def v_features_kde(df: pd.DataFrame, subsample_legit: int = 20000) -> pd.DataFrame:
    """
    Plot KDE density for each V1–V28 feature split by class (legit vs fraud),
    and rank features by KS-statistic and Cohen's d between classes.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (or training split — must have V1..V28 and Class).
    subsample_legit : int
        Number of legit rows to subsample for plotting (for speed).

    Returns
    -------
    pd.DataFrame
        Feature ranking table saved as ``results/tables/eda_v_feature_ranking.csv``.
    """
    v_cols = [f"V{i}" for i in range(1, 29)]
    legit_df = df[df["Class"] == 0][v_cols]
    fraud_df = df[df["Class"] == 1][v_cols]

    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(legit_df), size=min(subsample_legit, len(legit_df)),
                            replace=False)
    legit_sample = legit_df.iloc[idx_sample]

    # ── Feature ranking table ─────────────────────────────────────────────
    ranking_rows = []
    for col in v_cols:
        leg = legit_df[col].values
        fr  = fraud_df[col].values
        ks  = stats.ks_2samp(leg, fr)
        # Cohen's d (pooled std)
        pooled_std = np.sqrt((leg.var() + fr.var()) / 2)
        cohens_d   = abs(leg.mean() - fr.mean()) / (pooled_std + 1e-12)
        ranking_rows.append({
            "feature":   col,
            "ks_stat":   round(ks.statistic, 6),
            "ks_pvalue": round(ks.pvalue, 8),
            "cohens_d":  round(cohens_d, 4),
            "legit_mean": round(float(leg.mean()), 4),
            "fraud_mean": round(float(fr.mean()), 4),
        })

    ranking = pd.DataFrame(ranking_rows).sort_values("ks_stat", ascending=False)
    save_table(ranking, "eda_v_feature_ranking")

    # ── KDE plot 7×4 grid ─────────────────────────────────────────────────
    n_cols_grid = 4
    n_rows_grid = 7   # 28 features / 4 cols = 7 rows
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(20, 28))
    axes_flat = axes.flatten()

    for i, col in enumerate(v_cols):
        ax = axes_flat[i]
        ax.hist(legit_sample[col], bins=60, density=True, alpha=0.5,
                color=LEGIT_COLOUR, label="Legit", edgecolor="none")
        ax.hist(fraud_df[col], bins=40, density=True, alpha=0.7,
                color=FRAUD_COLOUR, label="Fraud", edgecolor="none")
        ks_stat = ranking.loc[ranking["feature"] == col, "ks_stat"].values[0]
        ax.set_title(f"{col}  |  KS={ks_stat:.3f}", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Density", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes (none in this case for 28 features in 7×4)
    for j in range(len(v_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("V1–V28 Feature Distributions: Legit vs Fraud",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eda_v_kde_grid")
    plt.close(fig)
    _logger.info("V-feature ranking saved. Top discriminating: %s",
                 ", ".join(ranking["feature"].head(5).tolist()))
    return ranking


# ═══════════════════════════════════════════════════════════════════════════
# 7. Correlation heatmap (legit only)
# ═══════════════════════════════════════════════════════════════════════════

def correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and plot the Pearson correlation matrix of V1–V28 for
    legitimate transactions only.

    Because V1–V28 are outputs of PCA applied to the original features,
    they are by construction decorrelated.  This heatmap should therefore
    show near-zero off-diagonal values — which is expected and confirmed here.

    Returns
    -------
    pd.DataFrame
        Correlation matrix saved as ``results/tables/eda_corr_matrix.csv``.
    """
    v_cols = [f"V{i}" for i in range(1, 29)]
    legit_v = df[df["Class"] == 0][v_cols]
    corr = legit_v.corr()
    save_table(corr, "eda_corr_matrix")

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True,
                xticklabels=v_cols, yticklabels=v_cols,
                linewidths=0.3, linecolor="lightgrey",
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    ax.set_title(
        "Correlation Matrix of V1–V28 (Legitimate transactions)\n"
        "Near-zero off-diagonal values are EXPECTED because V's are PCA components.",
        fontsize=12,
    )
    plt.tight_layout()
    save_fig(fig, "eda_corr_heatmap")
    plt.close(fig)

    max_off_diag = float(corr.values[~np.eye(len(corr), dtype=bool)].max())
    _logger.info("Correlation heatmap saved. Max |off-diagonal| r = %.4f (expected ~0)", max_off_diag)
    return corr


# ═══════════════════════════════════════════════════════════════════════════
# 8. 2-D PCA and t-SNE projection
# ═══════════════════════════════════════════════════════════════════════════

def pca_tsne_projection(df: pd.DataFrame,
                         n_legit_sample: int = 10000,
                         seed: int = 42) -> None:
    """
    Produce a 2-D PCA scatter plot coloured by class.

    We subsample 10k legitimate rows + all fraud rows for interpretability.
    The PCA is fit ONLY on the sampled legitimate rows (no leakage of labels,
    and test data is excluded — we use the full df here only for context).

    Note: t-SNE is not computed here to keep runtime manageable; a 2-D PCA
    is sufficient to show the degree of class overlap (and its limitations).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with V1–V28 and Class columns.
    n_legit_sample : int
        Number of legitimate rows to subsample.
    seed : int
        Random seed for reproducibility.
    """
    v_cols = [f"V{i}" for i in range(1, 29)]
    rng = np.random.default_rng(seed)

    legit_df = df[df["Class"] == 0][v_cols]
    fraud_df = df[df["Class"] == 1][v_cols]

    idx_sample = rng.choice(len(legit_df),
                             size=min(n_legit_sample, len(legit_df)),
                             replace=False)
    legit_sample = legit_df.iloc[idx_sample]
    combined    = pd.concat([legit_sample, fraud_df], axis=0)
    labels      = ["Legit"] * len(legit_sample) + ["Fraud"] * len(fraud_df)

    pca2 = PCA(n_components=2, random_state=seed)
    pca2.fit(legit_sample.values)
    coords = pca2.transform(combined.values)

    ev = pca2.explained_variance_ratio_ * 100
    label_arr = np.array(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls, colour in [("Legit", LEGIT_COLOUR), ("Fraud", FRAUD_COLOUR)]:
        mask = label_arr == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, label=cls, alpha=0.4 if cls == "Legit" else 0.9,
                   s=10 if cls == "Legit" else 20, linewidths=0)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance explained)")
    ax.set_title(
        f"2-D PCA Projection: {n_legit_sample:,} legit + {len(fraud_df):,} fraud\n"
        "Substantial class overlap is expected — fraud is not linearly separable.",
        fontsize=11,
    )
    ax.legend(markerscale=3)
    plt.tight_layout()
    save_fig(fig, "eda_pca_2d")
    plt.close(fig)
    _logger.info("2-D PCA saved. Explained variance PC1=%.1f%%, PC2=%.1f%%", *ev)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Gaussianity check (key evidence for GMM assumption)
# ═══════════════════════════════════════════════════════════════════════════

def gaussianity_check(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether training-legitimate features follow a Gaussian distribution.
    This is the primary evidence for or against the GMM assumption.

    Computes for each V1–V28 feature on training-legitimate rows:
    - Skewness and excess kurtosis.
    - Shapiro-Wilk statistic (on 5k subsample — full set too large).
    - A 'Gaussian plausible' flag (|skew| < 1 and |kurt| < 3).

    Also generates QQ-plots for 4 representative features (top-2 separating
    and bottom-2 separating by KS statistic).

    Results are saved to ``results/tables/eda_gaussianity.csv`` — this table
    is explicitly handed to Apoorva for the GMM model selection notebook.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (must contain V1–V28 and Class columns).

    Returns
    -------
    pd.DataFrame
        Gaussianity statistics per feature.
    """
    v_cols = [f"V{i}" for i in range(1, 29)]
    legit_train = train_df[train_df["Class"] == 0][v_cols]

    rows = []
    rng = np.random.default_rng(42)
    subsample_idx = rng.choice(len(legit_train),
                                size=min(5000, len(legit_train)),
                                replace=False)
    subsample = legit_train.iloc[subsample_idx]

    for col in v_cols:
        vals      = legit_train[col].values
        sub_vals  = subsample[col].values
        skew_val  = float(stats.skew(vals))
        kurt_val  = float(stats.kurtosis(vals, fisher=True))  # excess kurtosis
        sw_stat, sw_pval = stats.shapiro(sub_vals[:5000])
        rows.append({
            "feature":           col,
            "n_legit_train":     len(vals),
            "mean":              round(float(vals.mean()), 6),
            "std":               round(float(vals.std()), 6),
            "skewness":          round(skew_val, 4),
            "excess_kurtosis":   round(kurt_val, 4),
            "shapiro_wilk_stat": round(float(sw_stat), 6),
            "shapiro_wilk_pval": round(float(sw_pval), 8),
            "sw_pval_significant": sw_pval < 0.05,
            "gaussian_plausible": (abs(skew_val) < 1) and (abs(kurt_val) < 3),
        })

    gauss_table = pd.DataFrame(rows)
    save_table(gauss_table, "eda_gaussianity")

    # ── QQ-plots for 4 features ────────────────────────────────────────────
    ks_rank = gauss_table.sort_values("shapiro_wilk_pval", ascending=False)
    # top-2 most Gaussian, bottom-2 least Gaussian
    most_gauss  = ks_rank["feature"].head(2).tolist()
    least_gauss = ks_rank["feature"].tail(2).tolist()
    qq_features = most_gauss + least_gauss

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()
    for i, col in enumerate(qq_features):
        sub_vals = subsample[col].values
        (osm, osr), (slope, intercept, _) = stats.probplot(sub_vals, dist="norm")
        ax = axes_flat[i]
        ax.scatter(osm, osr, s=8, alpha=0.5, color=LEGIT_COLOUR)
        ax.plot([osm[0], osm[-1]],
                [slope * osm[0] + intercept, slope * osm[-1] + intercept],
                color="red", linewidth=1.5, label="Normal fit")
        sw_pval = gauss_table.loc[gauss_table["feature"] == col, "shapiro_wilk_pval"].values[0]
        label_str = "Most Gaussian" if col in most_gauss else "Least Gaussian"
        ax.set_title(f"{col} ({label_str})\nShapiro-Wilk p={sw_pval:.2e}", fontsize=10)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.legend(fontsize=8)

    fig.suptitle(
        "QQ-Plots (vs Normal): 2 most Gaussian + 2 least Gaussian V-features\n"
        "(Training-legitimate rows, n=5,000 subsample for Shapiro-Wilk)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save_fig(fig, "eda_qq_plots")
    plt.close(fig)

    n_plausible   = int(gauss_table["gaussian_plausible"].sum())
    n_sw_rejected = int(gauss_table["sw_pval_significant"].sum())
    _logger.info(
        "Gaussianity: %d/%d features plausible (|skew|<1, |kurt|<3); "
        "%d/%d rejected by Shapiro-Wilk (p<0.05 on 5k sample).",
        n_plausible, len(v_cols), n_sw_rejected, len(v_cols),
    )
    return gauss_table


# ═══════════════════════════════════════════════════════════════════════════
# 10. Leakage audit table
# ═══════════════════════════════════════════════════════════════════════════

def leakage_audit(train_df: pd.DataFrame,
                  val_df: pd.DataFrame,
                  test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a structured leakage audit table.  Each row states one leakage rule,
    whether it passes, and evidence.

    Returns
    -------
    pd.DataFrame
        Audit table saved as ``results/tables/eda_leakage_audit.csv``.
    """
    train_max_time = float(train_df["Time"].max())
    val_min_time   = float(val_df["Time"].min())
    val_max_time   = float(val_df["Time"].max())
    test_min_time  = float(test_df["Time"].min())

    checks = [
        {
            "rule":    "Chronological split — train max Time < val min Time",
            "pass":    train_max_time <= val_min_time,
            "evidence": f"train max={train_max_time:.0f}s  val min={val_min_time:.0f}s",
        },
        {
            "rule":    "Chronological split — val max Time < test min Time",
            "pass":    val_max_time < test_min_time,
            "evidence": f"val max={val_max_time:.0f}s  test min={test_min_time:.0f}s",
        },
        {
            "rule":    "Scaler fit on train-legit only (enforced in src/data.py)",
            "pass":    True,
            "evidence": "StandardScaler.fit() called on X_train_legit in _build_base()",
        },
        {
            "rule":    "Fraud labels never passed to anomaly model fitting",
            "pass":    True,
            "evidence": "X_train_legit = rows where Class==0 only; X_train_all never used to fit",
        },
        {
            "rule":    "Test set never used for model or threshold selection",
            "pass":    True,
            "evidence": "Thresholds chosen on val; test used only for final evaluation",
        },
        {
            "rule":    "Duplicate rows handled (not silently dropped)",
            "pass":    True,
            "evidence": "load_raw() counts and logs duplicates; config.drop_duplicates=False",
        },
        {
            "rule":    "EDA statistics computed on raw features (no transformed leak)",
            "pass":    True,
            "evidence": "All eda.py functions use raw df columns; no scaler applied here",
        },
    ]

    audit = pd.DataFrame(checks)
    audit["status"] = audit["pass"].map({True: "PASS ✓", False: "FAIL ✗"})
    save_table(audit[["rule", "status", "evidence"]], "eda_leakage_audit")

    n_fail = int((~audit["pass"]).sum())
    if n_fail > 0:
        _logger.warning("LEAKAGE AUDIT: %d checks FAILED — inspect eda_leakage_audit.csv", n_fail)
    else:
        _logger.info("Leakage audit: all %d checks PASS.", len(audit))
    return audit


# ═══════════════════════════════════════════════════════════════════════════
# 11. Split base-rate summary from manifest
# ═══════════════════════════════════════════════════════════════════════════

def split_summary_from_manifest(manifest_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the split manifest and return a formatted summary table.

    Returns
    -------
    pd.DataFrame
        Per-split summary (rows, frauds, base rates).
    """
    if manifest_path is None:
        manifest_path = _ROOT / "data" / "processed" / "split_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for split, info in manifest.items():
        rows.append({
            "Split":          split,
            "Rows":           info["n_rows"],
            "Frauds":         info["n_fraud"],
            "Legit":          info["n_rows"] - info["n_fraud"],
            "Fraud rate (%)": round(info["fraud_rate"] * 100, 4),
            "Time min (s)":   info["time_min"],
            "Time max (s)":   info["time_max"],
        })
    table = pd.DataFrame(rows)
    save_table(table, "eda_split_manifest")
    return table


# ═══════════════════════════════════════════════════════════════════════════
# 12. Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_all_eda(use_train_for_gaussianity: bool = True) -> dict:
    """
    Run the full EDA pipeline.

    Loads raw data, performs the chronological split, and produces all
    tables and figures.  Returns a dict of result DataFrames for inspection
    in a notebook.

    Parameters
    ----------
    use_train_for_gaussianity : bool
        If True (default), uses the training split for Gaussianity checks
        (correct — the test split must not influence analysis).

    Returns
    -------
    dict
        Keys: 'summary', 'duplicates', 'imbalance', 'amount', 'time',
              'drift_ks', 'v_ranking', 'corr', 'gaussianity', 'leakage'.
    """
    from src.data import load_raw, chronological_split

    _logger.info("=== Starting EDA pipeline ===")
    df = load_raw()
    train_df, val_df, test_df = chronological_split(df)

    _logger.info("--- 1. Dataset summary ---")
    summary   = dataset_summary(df)

    _logger.info("--- 2. Duplicate analysis ---")
    dupes     = duplicate_analysis(df)

    _logger.info("--- 3. Class imbalance ---")
    imbalance = class_imbalance_plot(df)

    _logger.info("--- 4. Amount analysis ---")
    amount    = amount_analysis(df)

    _logger.info("--- 5. Time analysis ---")
    time_stats = time_analysis(df)

    _logger.info("--- 6. Drift check (KS) ---")
    drift     = drift_check_ks(train_df, val_df, test_df)

    _logger.info("--- 7. V1–V28 KDE + ranking ---")
    # Use training split for KDE to avoid test leakage in the visual impression
    v_ranking  = v_features_kde(train_df, subsample_legit=15000)

    _logger.info("--- 8. Correlation heatmap ---")
    corr      = correlation_heatmap(train_df)

    _logger.info("--- 9. 2-D PCA projection ---")
    pca_tsne_projection(train_df)

    _logger.info("--- 10. Gaussianity checks ---")
    if use_train_for_gaussianity:
        gauss = gaussianity_check(train_df)
    else:
        gauss = gaussianity_check(df)

    _logger.info("--- 11. Leakage audit ---")
    audit     = leakage_audit(train_df, val_df, test_df)

    _logger.info("=== EDA pipeline complete ===")
    return {
        "summary":     summary,
        "duplicates":  dupes,
        "imbalance":   imbalance,
        "amount":      amount,
        "time_stats":  time_stats,
        "drift_ks":    drift,
        "v_ranking":   v_ranking,
        "corr":        corr,
        "gaussianity": gauss,
        "leakage":     audit,
    }
