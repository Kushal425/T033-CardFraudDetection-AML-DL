# Viva Notes — Krishna Pramod Dubale
## Phase 1: Evaluation Toolkit & Threshold Analysis (Task 04)

---

## 1. Why do we use PR-AUC instead of accuracy?

**Plain English**: The dataset has only 0.17% fraud. If a model always says "not fraud", it gets 99.83% accuracy — but catches zero fraud. Accuracy is useless here because true negatives (correctly saying "not fraud" for the vast majority) completely dominate the number.

PR-AUC (Precision-Recall Area Under Curve, also called Average Precision) measures how well the model ranks frauds above legitimate transactions. A random model gets PR-AUC ≈ 0.0017 (the base rate), while a perfect model gets 1.0. PR-AUC isn't inflated by the huge number of true negatives, so it honestly reflects detection ability.

We also report ROC-AUC for completeness, but it can be misleadingly high under imbalance because even many false positives translate to a tiny FPR when the denominator (all legitimate transactions) is enormous.

---

## 2. Bootstrap Confidence Interval — step by step

**What is it?** A way to estimate how uncertain our metric is, given the finite test data.

**Step-by-step procedure:**

1. Start with the test set: N=56,962 transactions, of which 75 are fraud.
2. Separate all indices into two pools: fraud indices (75) and legitimate indices (56,887).
3. Repeat B=2,000 times:
   - Sample 75 fraud indices **with replacement** from the 75 fraud indices.
   - Sample 56,887 legitimate indices **with replacement** from the 56,887 legitimate indices.
   - Combine them into a bootstrap replicate.
   - Compute the metric (e.g. PR-AUC) on this replicate.
4. Sort all 2,000 metric values.
5. The 95% CI is [2.5th percentile, 97.5th percentile].

**Why stratified?** If we resampled all indices together, some replicates might have 0 fraud rows (making PR-AUC undefined). Stratified resampling guarantees each replicate has exactly the original class ratio.

**Why are test CIs wide?** Because there are only 75 fraud transactions. Each bootstrap replicate draws 75 with replacement, so sometimes it gets 60 unique frauds, sometimes 85 draws from the 75 pool. This variability in the positive class makes the metric fluctuate a lot.

---

## 3. Recall@FPR=0.5% — in plain words

**Question**: "If we allow at most 0.5% of legitimate transactions to be falsely flagged, what fraction of actual frauds will we catch?"

**How we compute it**:
- On the validation set, look at the scores of legitimate transactions only.
- Find the 99.5th percentile of those scores — this becomes our threshold.
- A transaction is flagged if its score ≥ this threshold.
- On the test set, apply this threshold and count how many of the 75 frauds are above it.
- recall@FPR=0.5% = (frauds caught) / 75.

**Example interpretation**: If recall@FPR=0.5% = 0.80, it means the model catches 80% of frauds while falsely flagging only 0.5% of legitimate transactions (about 284 false alarms out of 56,887 legitimate test transactions).

---

## 4. Precision at Review Capacity

**Plain English**: The fraud investigation team can only review a limited number of flagged transactions per day. If they can review 100 transactions, we want to know: "Of the 100 most suspicious transactions, how many are actually fraud?"

- **Precision@K=100**: If 40 of the top-100 scored transactions are fraud, precision@100 = 0.40.
- **Precision@top-1%**: Same idea but the review budget is 1% of all transactions.

This is the **operational view** — it directly tells the team how productive their review time will be.

---

## 5. Why Thresholds Come from Validation, Not Test

**The rule**: All thresholds (FPR quantiles, max-F1, review capacity cutoffs) are derived from the validation set and then **frozen** — they are never adjusted based on test results.

**Why**: If we tuned thresholds on the test set, we'd be optimising for the specific test data. The reported performance would be overly optimistic because the model was "peeking" at the test answers. This is called **data leakage**.

**Our protocol**:
1. Compute thresholds from validation data only.
2. Lock them in — no further changes.
3. Apply the locked thresholds to test data and report the results.

Test labels are used **only** for computing evaluation metrics, never for any decision about models or thresholds.

---

## 6. What the Cost Curve Assumes

**ASSUMPTION**: A missed fraud costs **r** times as much as a false alarm.

We don't know the actual costs — they depend on the business (average transaction amount, investigation cost, insurance coverage, etc.). So we sweep r ∈ {10, 25, 50, 100, 250, 500} to show how the optimal threshold changes.

**How it works**:
- Expected cost per transaction = (fraction of missed frauds) × r + (fraction of false alarms) × 1
- For each r, we find the threshold on validation that minimises this cost.
- We then apply that threshold to the test set.

**Key insight**: Higher r → lower threshold → more aggressive flagging → fewer missed frauds but more false alarms. The "right" r is a business decision, not a statistical one.

---

## 7. Are Differences Between Models Significant?

**Method**: Paired bootstrap test.

1. Take the test set scores from model A and model B.
2. Resample both using the **same** bootstrap indices (paired).
3. Compute metric(A) − metric(B) for each of 2,000 resamples.
4. Construct a 95% CI of the difference.
5. If the CI excludes zero → the difference is **statistically significant**.
6. If the CI includes zero → the models are **statistically indistinguishable** given our data.

**Why paired?** Using the same resampled indices for both models controls for the random variation in which fraud cases appear in each replicate. This gives us a more precise estimate of the *difference* than comparing two independent CIs.

**Interpretation**: "Indistinguishable" does NOT mean the models are equally good in reality — it means our test set (with only 75 frauds) is too small to confidently distinguish them.

---

## 8. Why Test Results Have Wide CIs

The test set has 56,962 transactions but only **75 frauds** (0.13%). For fraud-sensitive metrics:

- Each bootstrap replicate draws 75 fraud indices with replacement.
- Due to the birthday paradox, each replicate sees only ~48-50 unique fraud transactions (the rest are duplicates).
- Whether a particular tricky fraud case lands in the replicate can swing the metric substantially.
- With ~75 fraud points, the standard error of recall is inherently ~1/√75 ≈ 0.12 (12 percentage points).

This is **not a flaw** in our methodology — it's an honest reflection of the statistical uncertainty. Any claim of high precision (narrow CI) with only 75 positive cases would be misleading.

---

## 9. Calibration Note

Anomaly scores (negative log-likelihood, isolation depth, distance to decision boundary) are **not probabilities**. They are arbitrary numbers that rank transactions by suspicion.

We can optionally map them to pseudo-probabilities using:
- **Platt scaling**: Fit a logistic regression on val (score → probability).
- **Isotonic regression**: Fit a monotonic mapping on val.

**Caveat**: With only ~57 val frauds, these calibration methods may be unreliable. The reliability diagram shows how well-calibrated the pseudo-probabilities are.

---

## 10. Evaluation Decisions Summary

| Decision | Rationale |
|---|---|
| Headline metric: PR-AUC | Not inflated by true negatives |
| Report accuracy: NO | Misleading at 0.17% base rate |
| Bootstrap: stratified, B=2000 | Preserves class ratio, sufficient for stable CIs |
| Thresholds: val-only | Prevents test-set leakage |
| Cost ratio: swept, assumed | Business-dependent, not a fixed number |
| Significance: paired bootstrap | Controls for replicate-level variation |
| Calibration: optional, caveated | Too few positives for reliable calibration |

---

*These notes cover every point listed in the viva preparation section of the task prompt.*
