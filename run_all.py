"""
run_all.py
==========
Orchestrates the available Phase 1 stages for Team T033 — Card Fraud Triage.

Stages (Phase 1 only)
---------------------
  eda       → Prachee's EDA notebook (requires nbconvert)
  features  → Feature engineering by Apoorva (src/features.py)
  gmm       → GMM baseline by Apoorva (src/gmm_model.py)
  baselines → Classical baselines by Vani (src/baselines.py)
  eval      → Evaluation toolkit by Krishna (src/evaluation.py)
  final     → Final comparison and report assembly by Kushal (src/final.py)

Stages that are NOT present yet will be skipped with an INFO message — they
will be added as teammates merge their PRs.

Phase 2 stages (ae, hybrid, ablation) are intentionally EXCLUDED from this
script.  They belong to Phase 2 and must not be imported here.

Usage
-----
  python run_all.py               # run all available stages
  python run_all.py --stages eda features gmm   # run specific stages
  python run_all.py --dry-run     # print stages without executing
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from typing import Callable

# ── Stage registry: name → (module, function) ────────────────────────────
# Each stage is a callable that takes no arguments and runs its pipeline.
# Using importlib so missing modules simply cause a SKIP, not an ImportError.

_STAGE_MAP: list[tuple[str, str, str]] = [
    # (stage_name, module_path, function_name)
    ("scaffold",   "src.data",          "get_features"),   # sanity check
    ("features",   "src.features",      "run_feature_stage"),
    ("gmm",        "src.gmm_model",     "run_gmm_stage"),
    ("baselines",  "src.baselines",     "run_baseline_stage"),
    ("eval",       "src.evaluation",    "run_eval_stage"),
    ("final",      "src.final",         "run_final_stage"),
]

_PHASE1_STAGES = [name for name, _, _ in _STAGE_MAP]


def _try_run_stage(stage_name: str, module_path: str, fn_name: str) -> str:
    """
    Attempt to import *module_path* and call *fn_name*().

    Returns one of: 'OK', 'SKIP' (module not found), 'ERROR'.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name.upper()}")
    print(f"{'='*60}")
    t0 = time.time()

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        print(f"  [SKIP] Module '{module_path}' not yet available — "
              f"will be added when the relevant PR is merged.")
        return "SKIP"

    if not hasattr(mod, fn_name):
        print(f"  [SKIP] '{module_path}.{fn_name}' not implemented yet.")
        return "SKIP"

    try:
        fn: Callable = getattr(mod, fn_name)
        # Special case: scaffold stage runs get_features() and prints shape
        if stage_name == "scaffold":
            splits = fn()
            print(f"  [OK] Scaffold check: X_val.shape = {splits.X_val.shape}")
        else:
            fn()
        elapsed = time.time() - t0
        print(f"  [OK] Stage '{stage_name}' completed in {elapsed:.1f}s")
        return "OK"
    except Exception:
        print(f"  [ERROR] Stage '{stage_name}' raised an exception:")
        traceback.print_exc()
        return "ERROR"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 pipeline stages for Team T033."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=_PHASE1_STAGES,
        default=_PHASE1_STAGES,
        metavar="STAGE",
        help=f"Stages to run (default: all). Choices: {_PHASE1_STAGES}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stages that would run without executing them.",
    )
    args = parser.parse_args()

    requested = args.stages
    print(f"\nT033 Phase 1 run_all.py")
    print(f"Requested stages: {requested}")

    if args.dry_run:
        print("\n[DRY-RUN] Would execute these stages in order:")
        for s in requested:
            print(f"  • {s}")
        return

    summary: dict[str, str] = {}

    for stage_name, module_path, fn_name in _STAGE_MAP:
        if stage_name not in requested:
            continue
        status = _try_run_stage(stage_name, module_path, fn_name)
        summary[stage_name] = status

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for stage, status in summary.items():
        icon = {"OK": "✓", "SKIP": "–", "ERROR": "✗"}.get(status, "?")
        print(f"  {icon} {stage:<12}  {status}")

    n_errors = sum(1 for s in summary.values() if s == "ERROR")
    if n_errors:
        print(f"\n{n_errors} stage(s) failed. Fix errors before submitting.")
        sys.exit(1)
    else:
        print("\nAll available stages completed successfully.")


if __name__ == "__main__":
    main()
