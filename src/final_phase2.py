"""
src/final_phase2.py
===================
Phase 2 final master comparison and integration pipeline.
Team T033 / Kushal Sarkar (Phase 2 Lead Integration).

Loads pre-computed score arrays from results/scores/ for all 10 models:
  Phase 1: PCA, LOF, GMM, OC-SVM, Isolation Forest
  Phase 2: Standard AE, Denoising AE (DAE), Hybrid H1 (Latent GMM),
           Hybrid H2 (Score Fusion), Hybrid H3 (DAGMM-lite)

Produces:
  - results/tables/final_comparison_phase2.csv  (master 10-model headline table)
  - results/tables/final_thresholds_phase2.csv (validation cutoffs and test workload)
  - results/tables/paper_numbers_p2.tex         (LaTeX macros for paper numbers)
  - results/figures/final_phase2_pr_curves.png   (all 10 models PR curves)
  - results/figures/final_phase2_recall_at_fpr.png (recall at operational FPRs)

Ranking metrics use stored scores. Threshold policies are fitted on validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import (
    base_rate,
    pr_auc,
    precision_at_fraction,
    precision_at_k,
    recall_at_fpr,
    recall_at_k,
    roc_auc,
)
from src.config import CFG
from src.data import chronological_split, load_raw
from src.evaluation import bootstrap_ci, threshold_policies_fpr

ALL_MODELS = [
    # Phase 1 classical baselines
    "pca",
    "lof",
    "gmm",
    "ocsvm",
    "iforest",
    # Phase 2 deep learning and hybrid models
    "ae",
    "ae_dae",
    "hybrid_latentgmm",
    "hybrid_fusion",
    "hybrid_dagmmlite",
]

MODEL_DISPLAY = {
    "pca": "P1: PCA Recon.",
    "lof": "P1: LOF",
    "gmm": "P1: GMM",
    "ocsvm": "P1: OC-SVM",
    "iforest": "P1: Iso. Forest",
    "ae": "P2: Deep AE",
    "ae_dae": "P2: Denoising AE",
    "hybrid_latentgmm": "P2: H1 Latent GMM",
    "hybrid_fusion": "P2: H2 Score Fusion",
    "hybrid_dagmmlite": "P2: H3 DAGMM-lite",
}

MODEL_COLORS = {
    "pca": "#1f77b4",          # strong blue
    "lof": "#2ca02c",          # green
    "gmm": "#ff7f0e",          # orange
    "ocsvm": "#9467bd",        # purple
    "iforest": "#d62728",      # red
    "ae": "#8c564b",           # brown
    "ae_dae": "#e377c2",       # pink / magenta
    "hybrid_latentgmm": "#7f7f7f",  # gray
    "hybrid_fusion": "#bcbd22",     # olive yellow
    "hybrid_dagmmlite": "#17becf",  # cyan
}


def load_labels() -> tuple[np.ndarray, np.ndarray]:
    """Load binary ground-truth labels for validation and test splits."""
    split_dir = ROOT / CFG["paths"]["processed"]
    val_path, test_path = split_dir / "val.csv", split_dir / "test.csv"
    if not val_path.is_file() or not test_path.is_file():
        _, val_df, test_df = chronological_split(load_raw())
    else:
        val_df = pd.read_csv(val_path, usecols=["Class"])
        test_df = pd.read_csv(test_path, usecols=["Class"])
    return val_df["Class"].values, test_df["Class"].values


def load_scores(split: str) -> dict[str, np.ndarray]:
    """Load 1-D anomaly score arrays for all 10 models for the specified split."""
    scores = {}
    for m in ALL_MODELS:
        path = ROOT / "results/scores" / f"{m}_{split}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing required score file: {path}")
        arr = np.load(path)
        scores[m] = arr
    return scores


def run_master_comparison() -> pd.DataFrame:
    """Compute complete comparative metrics across all 10 models and save results."""
    y_val, y_test = load_labels()
    s_val = load_scores("val")
    s_test = load_scores("test")
    for split, labels, scores in [("val", y_val, s_val), ("test", y_test, s_test)]:
        for model, values in scores.items():
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid {split} scores for {model}: expected {labels.shape}")

    fpr_budgets = CFG["eval"]["fpr_budgets"]
    cap_budgets = [0.001, 0.005, 0.010]
    k_reviews = [50, 100, 200]

    rows = []
    policies = []
    print("Computing metrics and bootstrap confidence intervals for all 10 models...")

    for m in ALL_MODELS:
        name = MODEL_DISPLAY[m]
        sv = s_val[m]
        st = s_test[m]

        v_prauc = pr_auc(y_val, sv)
        t_prauc = pr_auc(y_test, st)
        v_rocauc = roc_auc(y_val, sv)
        t_rocauc = roc_auc(y_test, st)

        # Bootstrap 95% CI for test PR-AUC (1000 resamples for speed and consistency)
        ci_res = bootstrap_ci(pr_auc, y_test, st, n=1000, seed=42)

        row = {
            "model_key": m,
            "Model": name,
            "val_pr_auc": round(float(v_prauc), 4),
            "val_roc_auc": round(float(v_rocauc), 4),
            "test_pr_auc": round(float(t_prauc), 4),
            "test_pr_auc_ci_lo": round(float(ci_res["ci_lo"]), 4),
            "test_pr_auc_ci_hi": round(float(ci_res["ci_hi"]), 4),
            "test_roc_auc": round(float(t_rocauc), 4),
        }

        # Test ROC points describe ranking. Validation thresholds simulate deployment.
        for fpr in fpr_budgets:
            pct_str = f"{fpr * 100:.1f}%"
            rec_val = recall_at_fpr(y_val, sv, fpr)
            rec_test = recall_at_fpr(y_test, st, fpr)
            row[f"val_recall_fpr_{pct_str}"] = round(float(rec_val), 4)
            row[f"test_recall_fpr_{pct_str}"] = round(float(rec_test), 4)
        policy = threshold_policies_fpr(y_val, sv, y_test, st, fpr_budgets)
        policy.insert(0, "model_key", m)
        policy["test_fpr"] = policy["FP"] / np.count_nonzero(y_test == 0)
        for result in policy.itertuples():
            pct_str = f"{result.fpr_budget:.1%}"
            row[f"test_recall_frozen_{pct_str}"] = round(result.recall, 4)
            row[f"test_fpr_frozen_{pct_str}"] = round(result.test_fpr, 6)
        policies.append(policy)

        # Precision at review capacities
        for cap in cap_budgets:
            pct_str = f"{cap * 100:.1f}%"
            p_cap = precision_at_fraction(y_test, st, cap)
            row[f"test_prec_cap_{pct_str}"] = round(float(p_cap), 4)

        # Precision at fixed top-K alerts
        for k in k_reviews:
            pk = precision_at_k(y_test, st, k)
            rk = recall_at_k(y_test, st, k)
            row[f"test_prec_k_{k}"] = round(float(pk), 4)
            row[f"test_recall_k_{k}"] = round(float(rk), 4)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    out_csv = ROOT / "results/tables/final_comparison_phase2.csv"
    df.to_csv(out_csv, index=False)
    pd.concat(policies, ignore_index=True).to_csv(
        ROOT / "results/tables/final_thresholds_phase2.csv", index=False
    )
    print(f"[final_phase2] Saved {out_csv} ({len(df)} models)")

    # ── Figure 1: PR Curves ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6.5))
    br = float(y_test.mean())
    ax.axhline(br, color="gray", linestyle="--", linewidth=1.2,
               label=f"Random Reference (Base Rate = {br:.4%})")

    # Order curves by test PR-AUC
    for m in df["model_key"]:
        st = s_test[m]
        prec, rec, _ = precision_recall_curve(y_test, st)
        auc_val = pr_auc(y_test, st)
        lw = 2.4 if m in ["pca", "ae_dae", "hybrid_dagmmlite"] else 1.5
        ls = "-" if m.startswith("ae") or m.startswith("hybrid") else "--"
        ax.plot(rec, prec, label=f"{MODEL_DISPLAY[m]} (AUC = {auc_val:.4f})",
                color=MODEL_COLORS[m], linewidth=lw, linestyle=ls)

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.set_title("Precision-Recall Curves — Phase 1 & Phase 2 Models (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    pr_fig_path = ROOT / "results/figures/final_phase2_pr_curves.png"
    fig.savefig(pr_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[final_phase2] Saved {pr_fig_path}")

    # ── Figure 2: Recall at FPR Budgets ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    width = 0.26

    for index, fpr in enumerate(fpr_budgets):
        target = f"{fpr:.1%}"
        offset = index - (len(fpr_budgets) - 1) / 2
        ax.bar(x + width * offset, df[f"test_recall_frozen_{target}"], width,
               label=f"{target} validation FPR target")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m].replace("P1: ", "").replace("P2: ", "") for m in df["model_key"]],
                       rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Recall on Test Frauds (N=75)", fontsize=11)
    ax.set_title("Test Recall at Thresholds Fixed on Validation", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    rec_fig_path = ROOT / "results/figures/final_phase2_recall_at_fpr.png"
    fig.savefig(rec_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[final_phase2] Saved {rec_fig_path}")

    return df


def write_paper_numbers_p2(df: pd.DataFrame):
    """Write LaTeX \\newcommand macros for all Phase 2 headline experimental numbers."""
    y_val, y_test = load_labels()
    s_val = load_scores("val")
    s_test = load_scores("test")

    br_test = base_rate(y_test)
    br_val = base_rate(y_val)

    lines = [
        "% Auto-generated by src/final_phase2.py — DO NOT EDIT BY HAND",
        "% Author: Kushal Sarkar (Phase 2 Lead Integration)",
        "% All values computed directly from verified results/scores/*.npy files",
        f"\\newcommand{{\\BaseRateTest}}{{{br_test * 100:.4f}\\%}}",
        f"\\newcommand{{\\BaseRateVal}}{{{br_val * 100:.4f}\\%}}",
        "\\newcommand{\\NTest}{56{,}962}",
        "\\newcommand{\\NVal}{56{,}961}",
        "\\newcommand{\\NFraudTest}{75}",
        "\\newcommand{\\NFraudVal}{57}",
    ]

    macro_names = {
        "pca": "PCA",
        "lof": "LOF",
        "gmm": "GMM",
        "ocsvm": "OCSVM",
        "iforest": "IFOREST",
        "ae": "AE",
        "ae_dae": "DAE",
        "hybrid_latentgmm": "HOne",
        "hybrid_fusion": "HTwo",
        "hybrid_dagmmlite": "HThree",
    }

    for m in ALL_MODELS:
        mac = macro_names[m]
        sv = s_val[m]
        st = s_test[m]

        prauc_v = pr_auc(y_val, sv)
        prauc_t = pr_auc(y_test, st)
        rocauc_t = roc_auc(y_test, st)
        rec_01 = recall_at_fpr(y_test, st, 0.001)
        rec_05 = recall_at_fpr(y_test, st, 0.005)
        rec_10 = recall_at_fpr(y_test, st, 0.010)
        p_cap05 = precision_at_fraction(y_test, st, 0.005)
        p_k100 = precision_at_k(y_test, st, 100)

        lines += [
            f"\\newcommand{{\\PRAUCTest{mac}}}{{{prauc_t:.4f}}}",
            f"\\newcommand{{\\PRAUCVal{mac}}}{{{prauc_v:.4f}}}",
            f"\\newcommand{{\\ROCAUCTest{mac}}}{{{rocauc_t:.4f}}}",
            f"\\newcommand{{\\RecallFPRTight{mac}}}{{{rec_01:.4f}}}",
            f"\\newcommand{{\\RecallFPRHalfPct{mac}}}{{{rec_05:.4f}}}",
            f"\\newcommand{{\\RecallFPROnePct{mac}}}{{{rec_10:.4f}}}",
            f"\\newcommand{{\\PrecCapHalfPct{mac}}}{{{p_cap05:.4f}}}",
            f"\\newcommand{{\\PrecKHundred{mac}}}{{{p_k100:.4f}}}",
        ]

    out_tex = ROOT / "results/tables/paper_numbers_p2.tex"
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[final_phase2] Saved {out_tex}")

    # Also copy to reports/ieee_report/paper_numbers_p2.tex for local include convenience
    local_tex = ROOT / "reports/ieee_report/paper_numbers_p2.tex"
    local_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("=" * 65)
    print("  PHASE 2 MASTER INTEGRATION & EMPIRICAL COMPARISON")
    print("=" * 65)
    df = run_master_comparison()
    write_paper_numbers_p2(df)
    print("=" * 65)
    print("  PHASE 2 MASTER COMPARISON COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
