# Viva Notes — Vani Rudra (Task 03: Classical ML Baselines)

## Overview

I implemented four classical anomaly-detection baselines plus two reference
baselines for fair comparison against the GMM density model.  All models are
trained on **legitimate training transactions only** — fraud labels are used
only for hyperparameter/threshold selection on the validation split and for
final evaluation on the held-out test split.

---

## 1. Isolation Forest

### What it does
Isolation Forest builds an ensemble of random binary trees.  Each tree
randomly selects a feature and a split value; it recursively partitions the
data until every point is "isolated" (alone in its partition).

### Key intuition
**Anomalies are few and different**, so they get isolated in fewer splits
(shorter path length).  Normal points are clustered and take many splits to
separate.

### Anomaly score
The score is based on the average path length across all trees:
```
s(x, n) = 2^(−E[h(x)] / c(n))
```
where `h(x)` is the path length for sample `x` and `c(n)` is the expected
path length for a balanced BST of `n` elements.  Scores close to 1 =
anomaly; close to 0.5 = normal.

In scikit-learn, `score_samples()` returns the negative of this anomaly
score, so I negate it: `score = −model.score_samples(X)`.

### Hyperparameters searched
- `n_estimators` ∈ {100, 200, 500} — number of trees
- `max_samples` ∈ {256, 1024, 'auto'} — how many samples per tree
- `max_features` ∈ {0.5, 1.0} — fraction of features per tree

### Why I did NOT use `contamination` for fitting
The `contamination` parameter only affects the hard threshold for
`predict()`.  I use `score_samples()` directly to get continuous scores, so
contamination is irrelevant and set to `'auto'`.

### Limitations
- Axis-parallel splits: IF cannot capture diagonal correlations between
  features.  If anomalies differ from normal data only along a rotated axis,
  IF needs many splits.
- Irrelevant features dilute the signal — each tree randomly selects
  features, so many irrelevant features waste splits.
- No density estimate: IF tells us *how easy* a point is to isolate, not
  *how likely* it is.

---

## 2. PCA Reconstruction Error

### What it does
PCA finds the top-k eigenvectors of the covariance matrix of legitimate
training data.  For any new point, we project it onto these k directions and
reconstruct it.  The anomaly score is the squared distance between the
original point and its reconstruction:
```
score = ‖x − x̂‖² = ‖x − U_k · U_k^T · x‖²
```

### Key intuition
Normal transactions "live" in a low-dimensional subspace (they cluster on a
manifold).  Anomalies have significant energy in the *residual* directions
that PCA discards, so their reconstruction error is high.

### Connection to Mahalanobis distance
The reconstruction error equals the squared Mahalanobis distance in the
minor (residual) subspace — when the residual eigenvalues are roughly equal.
This connects PCA reconstruction to probabilistic outlier detection.

### Hyperparameter: number of components (k)
I swept k from 1 to 31 (full dimensionality).  I also computed the k values
for 90%, 95%, and 99% explained variance.  The k with the best validation
PR-AUC was selected.

### Limitations
- **Linear**: PCA can only capture linear correlations.  If the normal data
  manifold is curved, PCA misses it.
- If an anomaly happens to project well into the top-k subspace (e.g., a
  high-value but otherwise normal transaction), PCA won't flag it.
- Sensitive to the choice of k: too many components → everything
  reconstructs well; too few → noisy reconstruction even for normal points.

---

## 3. One-Class SVM (RBF Kernel)

### What it does
OC-SVM maps data into a high-dimensional feature space using the RBF kernel:
```
K(x, x') = exp(−γ ‖x − x'‖²)
```
It then finds the maximum-margin hyperplane that separates the data from the
origin in this feature space.  Points on the "wrong side" (closer to the
origin) are anomalies.

### Anomaly score
`score = −decision_function(X)`.  The decision function is positive for
inliers, so negating gives higher scores for anomalies.

### Hyperparameters
- `ν` ∈ {0.001, 0.01, 0.05} — upper bound on fraction of outliers *and*
  lower bound on fraction of support vectors.  ν=0.01 means ≤1% of training
  data may be outside the boundary.
- `γ` ∈ {'scale', 0.01, 0.1} — RBF kernel width.  Smaller γ = smoother
  boundary.

### Why I sub-sample
OC-SVM has **O(n²·d) to O(n³·d)** fitting complexity because it needs the
full kernel matrix.  With 170,524 legitimate training rows, this would take
hours or days.  I use a seeded random sub-sample of **40,000 rows**, which
keeps fitting time manageable (minutes) while preserving enough data for a
good boundary estimate.

### Limitations
- Very sensitive to γ and ν; wrong values can make it flag everything or
  nothing.
- The RBF kernel implicitly assumes the data clusters in a roughly spherical
  region in feature space.
- Scales poorly to large datasets (even with sub-sampling).
- No density estimate — only a decision boundary.

---

## 4. Local Outlier Factor (LOF)

### What it does
LOF computes a *local reachability density* for each point based on its k
nearest neighbours.  The LOF score is the ratio of a point's density to the
average density of its neighbours.

### Key intuition
A point surrounded by a dense cluster but sitting at the edge will have a
*lower* density than its neighbours, giving it a high LOF score.  Unlike
global methods, LOF detects anomalies relative to their *local* context —
a point might be normal in one part of the feature space but anomalous in
another.

### Anomaly score
`score = −model.score_samples(X)`.  Scikit-learn's LOF in novelty mode
returns negative scores for outliers, so negating gives higher = more
anomalous.

### Why `novelty=True`
Without `novelty=True`, LOF can only label the training data — it cannot
score new points.  With `novelty=True`, the model is fit on training data
and then used to score unseen validation/test data.

### Hyperparameter
- `n_neighbors` ∈ {10, 20, 50} — how many neighbours define the "local"
  context.

### Sub-sampling
Like OC-SVM, LOF's fitting cost involves computing distances between all
pairs (O(n·k·d) plus building a k-d tree or ball tree).  I sub-sample to
40,000 rows with a fixed seed.

### Limitations
- In high dimensions, distances become less meaningful (curse of
  dimensionality).  All points appear roughly equidistant, so LOF scores
  converge.
- Sensitive to k: too small → noise; too large → over-smoothing.
- Cannot handle datasets where anomalies cluster together (LOF detects
  points that are locally sparse, not globally rare).

---

## 5. Random Baseline (Sanity Check)

### What it does
Assigns uniform random scores U(0,1) to each transaction.

### Why it matters
A random ranker's PR-AUC should equal the **base rate** (fraction of
frauds in the split, ≈0.1%).  This is a critical sanity check:
- If a model's PR-AUC is not meaningfully above the random baseline, the
  model is no better than guessing.
- If the random baseline's PR-AUC is far from the base rate, something is
  wrong with the evaluation pipeline.

---

## 6. |Amount| Baseline

### What it does
Uses the absolute (scaled) transaction amount as the anomaly score.

### Why it matters
Tests a simple hypothesis: "are high-value transactions more likely to be
fraudulent?"  If this simple feature alone has high PR-AUC, it tells us
amount is a strong fraud signal.  If not, it confirms that fraud patterns
are more complex than just "big transactions."

---

## Hyperparameter Selection Protocol

**Rule (stated before running any experiments):** For each model, I select
the hyperparameters that maximise **validation PR-AUC**.

Why PR-AUC and not ROC-AUC?
- The dataset has severe class imbalance (fraud ≈ 0.1%).
- ROC-AUC is inflated by the large number of true negatives.
- PR-AUC focuses on the positive (fraud) class and is more informative
  for imbalanced problems.

I never touch the test set for model selection.  The test set is evaluated
exactly once with the best model from validation.

---

## Score Direction Convention

All scores follow: **higher = more anomalous = more likely fraud**.

- PCA: higher reconstruction error = more anomalous ✓
- IF: negated score_samples (sklearn uses the opposite convention) ✓
- OC-SVM: negated decision_function (positive = inlier in sklearn) ✓
- LOF: negated score_samples ✓
- Random: uniform [0,1] — direction is irrelevant for random ✓
- Amount: higher amount = more suspicious (heuristic) ✓

---

## Implementation Decisions

1. **Feature set**: I use `get_features("base")` which gives 31 features:
   V1–V28, log1p(Amount), sin(time), cos(time).  All standardised with a
   scaler fit on training-legitimate rows only.

2. **Sub-sample size for OC-SVM and LOF**: 40,000 rows.  This is in the
   range specified by the task (30k–50k).  I chose 40k because:
   - It's large enough to represent the data distribution well.
   - It keeps OC-SVM fitting time to minutes, not hours.
   - The seed is fixed (42) for reproducibility.

3. **PCA sweep range**: I sweep all k from 1 to 31 (the feature dimension).
   This is exhaustive and ensures I don't miss the optimal k.

4. **Score saving**: All scores are saved as float32 `.npy` arrays with
   lengths matching the split sizes (val=56,961; test=56,962).

5. **GMM comparison**: If GMM scores are available (from Apoorva's task),
   they are included in the comparison table and rank-correlation analysis.

---

## Which Baseline Was Strongest?

Based on actual experimental results (see `results/tables/baseline_metrics.csv`):

| Model          | Val PR-AUC | Test PR-AUC | Val ROC-AUC | Test ROC-AUC |
|----------------|-----------|------------|------------|-------------|
| **PCA (k=30)** | **0.7066** | **0.7425** | 0.9301     | 0.9303      |
| LOF (k=20)     | 0.4069    | 0.4478     | 0.9410     | 0.9048      |
| GMM            | 0.1950    | 0.4142     | 0.9453     | 0.9274      |
| OC-SVM         | 0.0613    | 0.1322     | 0.9572     | 0.9604      |
| IForest        | 0.0330    | 0.0577     | 0.9469     | 0.9567      |
| Amount         | 0.0033    | 0.0034     | 0.7357     | 0.7095      |
| Random         | 0.0011    | 0.0015     | 0.4598     | 0.4840      |

**PCA reconstruction error (k=30) was the clear winner** with val PR-AUC
of 0.7066 and test PR-AUC of 0.7425.  This is because:
- The credit card V1–V28 features are already PCA-transformed, so
  reconstruction error in the residual subspace is highly effective at
  detecting anomalies that deviate from the normal data manifold.
- With k=30 out of 31 features, the model captures nearly all variance
  (99.13%) and flags points with energy in the single residual dimension.
- The top-20 highest PCA scores on validation were **all true frauds**
  (20/20 precision), confirming strong signal separation.

**LOF (n_neighbors=20) was second** at 0.4069 val PR-AUC, showing that
local density estimation captures fraud patterns missed by global methods.

**Key surprise:** Isolation Forest performed poorly (0.033 val PR-AUC)
despite good ROC-AUC (0.947).  This ROC-AUC/PR-AUC disconnect is classic
in imbalanced data — ROC-AUC is inflated by true negatives, while PR-AUC
correctly penalises the many false positives.  IF's axis-parallel splits
struggle with the correlated PCA features.

**Sanity checks passed:**
- Random baseline PR-AUC ≈ 0.001 ≈ base rate (0.001) ✓
- All score directions verified (higher = more anomalous) ✓

---

## Future Work (one sentence)

In Phase 2, these classical baselines will be compared against deep learning
approaches (autoencoders) to determine whether learned nonlinear
representations provide a meaningful advantage over the classical methods.

---

*Prepared by Vani Rudra for T033 Phase 1 viva.*
