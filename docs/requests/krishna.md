# Change Requests — Krishna Pramod Dubale (Task 04)

This document lists changes that Krishna needs in shared files or other team
members' files.  Since I cannot edit shared files directly, these are requests
for Kushal (integration lead) or the relevant file owner.

---

## No blocking changes required

As of Phase 1 submission, the evaluation toolkit (`src/evaluation.py`) works
with the existing shared infrastructure:

- **`src/metrics.py`**: All required metric functions are already present.
  The evaluation module imports from `src.metrics` without modification.
- **`src/utils.py`**: `save_scores`, `load_scores`, `save_table`, `save_fig`
  all work as documented.
- **`configs/config.yaml`**: The `eval` section has the correct `fpr_budgets`,
  `review_fractions`, and `top_k` settings.

## Nice-to-have (non-blocking)

1. **`configs/config.yaml`** — Consider adding a `bootstrap` section:
   ```yaml
   bootstrap:
     n_resamples: 2000
     seed: 42
     alpha: 0.05
   ```
   Currently these are hardcoded in `src/evaluation.py`.  Moving them to
   config would make them easier to tune.  Not blocking: defaults are
   reasonable.

2. **`requirements.txt`** — `seaborn` should be added to the requirements:
   ```
   seaborn>=0.12
   ```
   It is listed in the plotting requirements comment but `src/evaluation.py`
   imports it (used for consistent plot styling).

3. **Training/inference timing data** — The master comparison table has
   columns for `train_time_s` and `inference_time_s`.  These are currently
   NaN because the GMM and baseline score files don't include timing
   metadata.  If Apoorva and Vani can record `time.time()` around training
   and inference, the evaluation table will be more complete.

---

*Last updated: Phase 1 submission.*
