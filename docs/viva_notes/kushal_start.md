# Viva Notes — Kushal Sarkar (Scaffold / Lead)
**Task 00 — P1_0A — Team T033, Project 26**

This document explains every function in the scaffold in plain English so I
can answer questions in the individual viva without referring to code.

---

## 1. `src/config.py`

### `get_repo_root()`
**What it does:** Returns the absolute path to the repository root folder.

**How:** This file (`config.py`) lives at `<repo>/src/config.py`. Python's
`__file__` gives us the current file's path. Going two levels up (`.parent.parent`)
from `src/config.py` gives us the repo root. Every other module can call
`get_repo_root()` instead of hard-coding paths.

**Why:** If someone clones the repo in a different location, the hard-coded
path `/Users/kushal/Desktop/...` would break. A relative-to-file approach works
anywhere.

### `load_config(path)`
**What it does:** Reads `configs/config.yaml` and returns a Python dictionary.

**How:** Opens the YAML file with `yaml.safe_load()` (which parses YAML into
Python dicts/lists). If the file is missing, raises a clear `FileNotFoundError`.

**Why:** Centralising config loading means changing `seed: 42` to `seed: 0`
in one file affects every module that imports `CFG`.

### `CFG`
**What it is:** A module-level dictionary created when `config.py` is first
imported. Contains all settings: seed, paths, split fractions, feature set,
eval budgets.

**Viva tip:** "Every module does `from src.config import CFG` — so there is
exactly ONE config object shared across the whole codebase."

---

## 2. `src/utils.py`

### `set_seed(seed)`
**What it does:** Sets the random seed for Python's built-in `random` module,
NumPy, and (optionally) PyTorch. Also sets `PYTHONHASHSEED` for dictionary
ordering.

**Why:** Machine learning experiments must be reproducible. Without fixing the
seed, running the same code twice can give different train/val splits, GMM
initialisations, or metric values. `seed=42` comes from `CFG['seed']`.

**How:** `random.seed(seed)` → Python; `np.random.seed(seed)` → NumPy;
`torch.manual_seed(seed)` → PyTorch (inside a try/except because PyTorch is
Phase 2-only).

### `get_logger(name, level)`
**What it does:** Returns a Python `logging.Logger` with a consistent format
(timestamp, level, module name, message).

**Why:** Using `print()` everywhere makes it hard to filter or redirect
output. Python's logging module lets you set levels (DEBUG, INFO, WARNING,
ERROR) and redirect logs to files in production.

**How:** `logging.getLogger(name)` is a registry — the same name always
returns the same logger object. We add a StreamHandler only if no handlers
are already attached (prevents duplicate messages).

### `save_scores(model_name, split, scores)`
**What it does:** Saves a 1-D NumPy float32 array to
`results/scores/{model_name}_{split}.npy`.

**Score contract:** Higher score = more anomalous = more likely to be fraud.
Length = number of rows in that split. Row order = same as `data/processed/{split}.csv`.

**Why this contract matters:** If Apoorva saves GMM scores and Krishna loads
them for the evaluation toolkit, they must agree on the file format.
A broken contract → wrong PR curves and incorrect threshold analysis.

**How:** `np.save()` writes a binary `.npy` file. It is fast and exact
(no floating-point text representation).

### `load_scores(model_name, split)`
**What it does:** Loads the `.npy` file saved by `save_scores`. Raises
`FileNotFoundError` with a helpful message if the model hasn't been run yet.

### `save_table(df, name)` / `save_fig(fig, name)`
**What they do:** Save pandas DataFrames as CSVs and matplotlib Figures as
PNGs to `results/tables/` and `results/figures/` respectively.

**Why separate functions?** Keeps all path logic in one place. Every module
calls `save_table(df, "gmm_metrics")` instead of building the path itself.

---

## 3. `src/data.py`

### `load_raw()`
**What it does:** Reads `data/raw/creditcard.csv`, validates the schema
(31 columns: Time, V1–V28, Amount, Class; all numeric; Class ∈ {0,1}),
logs duplicate row counts, and returns the raw DataFrame.

**Steps in order:**
1. Check file exists → raise `FileNotFoundError` with download instructions if not.
2. Read CSV with `pd.read_csv()`.
3. Assert all 31 expected columns are present.
4. Check all columns are numeric.
5. Check Class ∈ {0, 1}.
6. Count duplicate rows (`df.duplicated().sum()`) → log the count.
7. If `drop_duplicates: true` in config → drop them; otherwise keep + warn.
8. Log fraud base rate.

**Why validate schema?** If someone accidentally places the wrong CSV (e.g.,
a different dataset), the schema check immediately tells them what's wrong
instead of failing mysteriously inside the model.

**Why not drop duplicates silently?** Silent data modifications are hard to
notice and reproduce. The config flag + log message makes the decision explicit.

### `chronological_split(df)`
**What it does:** Sorts the DataFrame by `Time` (no shuffling), then cuts
at 60 % and 80 % of the total row count to create train, val, and test splits.
Saves CSVs and a JSON manifest.

**Key detail:** The sort uses `kind="stable"` so rows with identical `Time`
values preserve their original relative order.

**Why chronological?** In real-world fraud detection, the model is trained on
past transactions and predicts on future ones. A random split would mix future
into training — "leaking the future" into the model.

**Why save the manifest?** The JSON file records the exact row counts, fraud
counts, base rates, and Time min/max per split. This lets teammates (and the
faculty evaluator) verify the split without re-running the pipeline.

**The < 50 fraud warning:** If val or test has fewer than 50 fraud events,
threshold estimates become unstable and PR-AUC confidence intervals become
very wide. We warn loudly and suggest 60/15/25 if this happens.

**Time ordering guarantee:**
`max(train.Time) ≤ min(val.Time) ≤ max(val.Time) ≤ min(test.Time)`

### `_build_base(train_df, val_df, test_df)` (internal)
**What it does:** Constructs the "base" feature set from the split DataFrames.

**Features created:**
- `V1`–`V28`: taken directly (already numeric PCA components).
- `log1p_amount`: `log(1 + Amount)`. The log transform compresses the heavy
  tail of the Amount distribution (most transactions are small; some are very
  large). `log1p` is used instead of `log` to safely handle Amount=0.
- `sin_time`, `cos_time`: Cyclic encoding of `Time mod 86400` (seconds per day).
  Using sin/cos instead of a raw hour value preserves the cyclic nature
  (23:59 is close to 00:01). This is a common trick for periodic time features.

**ASSUMPTION documented here:** We assume Time=0 ≈ midnight. The dataset
does not confirm this, so the cyclic encoding is an approximation.

**Scaler:** `StandardScaler` is fit on training-legit rows only (Class=0 in
the training split). It subtracts the mean and divides by the standard
deviation computed from those rows. This ensures no information from fraud
rows or future splits contaminates the scaling parameters.

**Why legit-only?** If we fit the scaler on all rows including fraud, the
mean and variance would slightly reflect fraud patterns. More importantly,
fitting on the full training set (including some test-like fraud rows)
is conceptually a form of leakage.

### `FEATURE_SETS` registry
A Python dict mapping string names to builder functions:
```python
FEATURE_SETS = {"base": _build_base}
```
When Apoorva creates `src/features.py`, her function updates this dict
with `"engineered"`. A `try/except ImportError` block attempts the import
but never crashes if the file is missing.

### `get_features(feature_set)`
**What it does:** Calls `load_raw()`, `chronological_split()`, looks up the
feature builder in `FEATURE_SETS`, runs it, and returns a `Splits` dataclass.

**It is the single entry point.** All teammates call:
```python
from src.data import get_features
s = get_features()
```
They then use `s.X_train_legit`, `s.X_val`, `s.y_test`, etc.

### `Splits` dataclass
A container with eight arrays and two metadata fields:
- `X_train_legit` — scaled features for training legitimate rows only.
  **Fit your anomaly model on this.**
- `X_train_all` — scaled features for all training rows (fraud + legit).
  **For diagnostics only — NEVER fit a model on this.**
- `y_train_all`, `X_val`, `y_val`, `X_test`, `y_test` — self-explanatory.
- `feature_names` — list of feature column names (for plots and reports).
- `scaler` — the fitted `StandardScaler` object (for inverse-transforming,
  computing means, or feature importance in space of original units).

---

## 4. `src/metrics.py`

### Convention: higher score = more anomalous
Every function in this module assumes **higher score = more likely fraud**.
This is the team-wide contract. Violation = wrong PR curves.

### `base_rate(y)`
**What it does:** Returns the fraction of 1s (frauds) in the label array.
**Why:** The base rate ≈ 0.17 % sets the random-classifier PR-AUC baseline.
A model that's better than random must exceed this. Never report accuracy;
a model predicting "always legit" achieves 99.83 % accuracy.

### `pr_auc(y, s)`
**What it does:** Computes Average Precision = area under the Precision-Recall
curve.

**Why PR-AUC instead of ROC-AUC?**
ROC-AUC counts true negatives. With 99.83 % legitimate rows, even a terrible
model can have a high ROC-AUC because it handles the easy negatives well.
PR-AUC only cares about how well the model finds the rare positives (fraud).
A random ranker has PR-AUC ≈ 0.17 %; a perfect model has PR-AUC = 1.0.

### `roc_auc(y, s)`
**What it does:** Area under the ROC curve (TPR vs FPR). Reported for
completeness but must not be the headline metric.

### `recall_at_fpr(y, s, fpr)`
**What it does:** Given an FPR budget (e.g., 0.5 %), returns the highest
fraction of frauds we can catch while keeping false alarms below that budget.

**How:** Compute the full ROC curve (sklearn returns arrays of FPR and TPR).
Filter to operating points where FPR ≤ budget. Return max(TPR) among those.

**Why this metric matters operationally:** A fraud team can review N alerts
per day. If we allow 0.5 % of legitimate transactions to be false alarms,
how many real frauds do we catch? That's recall@FPR=0.5 %.

### `threshold_at_fpr(y_val, s_val, fpr)`
**What it does:** Derives a decision threshold from the validation legitimate
score distribution.

**How:** Take the (1 - fpr) quantile of the val-legit scores. If we set a
threshold at this value, approximately fpr × 100 % of val-legit transactions
would be flagged as false alarms.

**Why from val legit only?** The threshold must be set BEFORE looking at the
test set. Fraud rows are excluded so the threshold reflects the "normal" score
distribution, not the fraud density.

### `precision_at_k(y, s, k)` / `recall_at_k(y, s, k)`
**What they do:** Flag the top-k highest-scored transactions. Precision@k =
fraction of those k that are fraud. Recall@k = fraction of all frauds found.

**Operational meaning:** If a fraud team can review 100 cases per day
(k=100), what fraction of those reviews catch a real fraud (precision@100)?
And how many of all actual frauds do we find (recall@100)?

### `precision_at_fraction(y, s, frac)`
Reviews the top `frac × N` transactions. Equivalent to `precision_at_k` with
`k = round(N * frac)`.

### `f1_at_threshold(y, s, thresh)` / `confusion_at_threshold(y, s, thresh)`
Standard classification metrics applied at a fixed threshold. The confusion
matrix shows TP/FP/TN/FN; F1 = 2·TP / (2·TP + FP + FN).

### `evaluate_scores(y_val, s_val, y_test, s_test, cfg)`
**What it does:** Computes the full evaluation table for one model.

**Critical leakage rule enforced here:** Thresholds are derived from
`y_val, s_val` (validation data) and then **applied unchanged to test**.
Test labels are used ONLY to evaluate the outcome, never to select thresholds.

**Returns:** A DataFrame with rows = metric names, columns = ['val', 'test'].

---

## 5. Tests (`tests/`)

### `test_no_leakage.py`

**`test_time_ordering`**
Checks that `max(train.Time) ≤ min(val.Time)` and `max(val.Time) ≤ min(test.Time)`.
If this fails, future data has leaked into training (e.g., because someone
shuffled the DataFrame before splitting).

**`test_scaler_mean_equals_train_legit_mean`**
Reconstructs the unscaled training-legit features from `X_train_legit` and the
scaler parameters, then checks that the column means equal `scaler.mean_`.
If the scaler was accidentally fit on all rows (including fraud or future rows),
its mean would differ from the legit-only mean.

**`test_no_fraud_in_X_train_legit`**
Checks that `X_train_legit.shape[0]` equals the number of Class=0 rows in
the training split. If even one fraud row slips in, the count would be larger.

### `test_metrics.py`

**`test_perfect_scores_pr_auc_equals_one`**
A scorer that assigns score=1 to every fraud and score=0 to every legit row
is a perfect ranker. Its PR-AUC must be exactly 1.0.

**`test_random_scores_pr_auc_approx_base_rate`**
A uniformly random scorer is uncorrelated with labels. Its PR-AUC ≈ base_rate.
This tests that `pr_auc` is not returning a hard-coded value.

**`test_recall_at_fpr_monotone`**
As the FPR budget increases (we allow more false alarms), we can only catch
more (or equal) frauds — recall is monotonically non-decreasing in FPR.
A violation would indicate a bug in the ROC curve filtering.

---

## 6. Key viva questions I must be able to answer

1. **Why anomaly detection instead of supervised classification?**
   Fraud labels are expensive and the class is extremely rare (0.17 %).
   Anomaly detection trains only on the abundant legitimate data and flags
   transactions that look "different" from the learned normal distribution.

2. **What is leakage and where could it happen?**
   Leakage = using future or label information during training.
   (a) Scaler fit on full dataset → test statistics leak into training.
   (b) Shuffling before splitting → future transactions appear in training.
   (c) Fitting on fraud rows → the anomaly model is exposed to its targets.

3. **Why log1p(Amount)?**
   The Amount distribution is heavy-tailed (skewed right). A few very large
   transactions would dominate the variance and push the standardised scale.
   `log1p` compresses this tail, giving the StandardScaler a more Gaussian-like
   distribution to work with.

4. **Why sin/cos encoding for time?**
   A raw hour value treats 23 and 0 as far apart (23 → 0 = jump of 23 units)
   when they are actually 1 hour apart. Mapping to `(sin(2π·h/24), cos(2π·h/24))`
   on a unit circle makes the encoding continuous and cyclic.

5. **Why is PR-AUC the headline metric and not accuracy?**
   With 0.17 % fraud rate, a model that always predicts "legitimate" gets
   99.83 % accuracy. This is useless. PR-AUC measures how well the model
   distinguishes the rare positives from the overwhelming negatives.

6. **What does the split manifest tell us?**
   Row counts, fraud counts, base rates per split, and Time min/max. This lets
   anyone verify the split is correct and chronological without re-running the
   pipeline or committing the data.

7. **Why are thresholds chosen on val and applied to test without change?**
   If we peek at test to choose a threshold, we are optimising for the test
   set — the threshold would be over-fitted to test performance and the
   evaluation would be invalid. The test set must remain "unseen" until the
   very end.
