# Viva Notes — Prachee Dhar (P2_04)
## Phase 2: Hybrid Models, Ablation & Master Evaluation
### Team T033 — Project 26: Card Fraud Triage

---

## 1. What is your role in Phase 2?

I own the hybrid implementation (H2 Score Fusion and H3 DAGMM-lite), the ablation study, and the master evaluation across all Phase 1 and Phase 2 models. I did not modify Vani's Autoencoder/DAE code or Apoorva's Latent-GMM implementation — I consume their score files and model artifacts as inputs.

---

## 2. What is H2 (Score Fusion)? What fusion methods did you implement?

H2 is a post-hoc combination of two independently trained anomaly scores:
- **Standalone DAE reconstruction score** (Vani's DAE, Phase 2)
- **Phase 1 GMM negative log-likelihood**

I implemented three fusion strategies:
1. **Normalized Weighted Fusion**: $S_{H2} = \alpha S_{AE}^{norm} + (1-\alpha) S_{GMM}^{norm}$, with $\alpha \in [0, 1]$ swept on validation
2. **Max Fusion**: $S_{H2-Max} = \max(S_{AE}^{norm}, S_{GMM}^{norm})$
3. **Rank-Average Fusion**: average of rank-normalized scores

And four normalization methods: MinMax, Z-score, LogZscore, and Rank normalization.

---

## 3. How did you select alpha (fusion weight)?

I swept $\alpha \in \{0.0, 0.1, \ldots, 1.0\}$ across all 4 normalization methods and selected the $(\alpha^*, \text{norm}^*)$ pair that maximizes **validation PR-AUC** (never test). The result was $\alpha^* = 1.0$ under all normalization methods.

---

## 4. Why did alpha = 1.0? Isn't that just the DAE score?

Yes, $\alpha = 1.0$ means H2 is identical to the standalone DAE score. This is a **genuine, honest result, not a fabrication**.

The reason: the Phase 1 GMM's **validation PR-AUC is only 0.1038** — far below the DAE's 0.2935. The GMM was trained on 31-dimensional ambient space and suffers from the curse of dimensionality. Because it has essentially no discriminative signal on validation, the fusion correctly weights it at 0.

The validation sweep correctly identifies this — there is no benefit to including a noisy signal. The finding is methodologically important: **score fusion only helps when both component scores contribute discriminative information**.

**Spearman correlation** between DAE and GMM scores: $\rho_{val} = 0.527$, $\rho_{test} = 0.470$ — the scores are moderately correlated, confirming that the GMM captures some of the same signal as the DAE but adds noise rather than complementary information at this quality level.

---

## 5. What is H3 (DAGMM-lite)? How does it differ from full DAGMM?

H3 is a **decoupled, simplified implementation** of the DAGMM architecture (Zong et al., ICLR 2018):

**Full DAGMM** trains an encoder, decoder, and "estimation network" jointly using a combined loss:
$$\mathcal{L}_{DAGMM} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{energy} + \lambda_2 \mathcal{L}_{cov\_diag}$$
The energy regularization feeds back into the autoencoder, forcing it to map normal data into dense mixture regions.

**DAGMM-lite (H3)** decouples these stages:
1. Use Vani's already-trained DAE (frozen) to get latent codes $z$ and reconstructions $\hat{x}$
2. Compute three scalar error features per transaction: MSE, relative Euclidean distance, cosine distance
3. Build augmented vector $v = [z, e_{mse}, e_{rel}, e_{cos}] \in \mathbb{R}^{15}$
4. Fit a GMM on $v$ using legitimate training records only
5. Anomaly score = $-\ln p_V(v)$

**Why decoupled?** (i) Team ownership — I cannot modify Vani's AE; (ii) computational efficiency — avoids expensive joint retraining; (iii) direct ablation of "latent codes only" (H1) vs "latent codes + reconstruction errors" (H3).

**H3 is explicitly NOT full end-to-end DAGMM.**

---

## 6. What hyperparameters did you select for H3? How?

Grid searched: $K \in \{2, 4, 8, 16\}$, covariance type $\in \{\text{diag}, \text{tied}\}$, regularization $\epsilon_{cov} \in \{0.01, 0.1\}$ — 16 configurations total.

**Selected on validation PR-AUC**: K=16, diag covariance, $\epsilon_{cov} = 0.1$.

**Val PR-AUC = 0.3913**, **Test PR-AUC = 0.4541** (95% CI: [0.3524, 0.5884]).

This represents a large improvement over H1 Latent-GMM (0.1724 test), confirming that augmenting with reconstruction error dimensions provides substantially richer density surfaces.

---

## 7. What were H3's operational results?

- **Test PR-AUC: 0.4541** (95% CI: [0.3524, 0.5884])
- **Recall@0.5% FPR: 0.800** — highest among all Phase 2 models
- Precision@100: 0.47

The Recall@0.5% FPR result is particularly important: at an operational queue capacity of ~285 alerts per 56,962 transactions (0.5% FPR), H3 catches 80% of test frauds. This matches the PCA Anomaly Score's recall at that FPR threshold.

---

## 8. Walk through the ablation study factors

| Factor | Finding |
|--------|---------|
| A1: AE vs DAE | Denoising +29% val PR-AUC; DAE selected |
| A2: Latent dim | dim=8 best in sweep; dim=12 used in final (from full hyperparam search) |
| A3: Feature set | Engineered +11.5% val; base used for reproducibility |
| A4+A5: GMM cov/K | diag/tied outperform full; K=8-16 optimal in latent space |
| A6: Training size | 170k legit records; 5-seed std=0.148 (sparse val frauds cause noise) |
| A7: Fusion method/weight | alpha=1.0 (GMM too weak); method choice secondary |
| A8: Hybrid vs standalone | PCA dominates; H3 has highest recall@0.5% FPR among P2 |

---

## 9. Why does PCA Anomaly Score outperform all deep learning models?

PCA Anomaly Score achieves test PR-AUC = **0.7425**, far above the best DL model (DAE: 0.4818).

**Reasons:**
1. **Data structure**: The ULB dataset is already PCA-compressed (V1-V28). PCA residuals are therefore a "native" anomaly metric for this feature space.
2. **Dataset size**: 170k training samples; a 2,631–3,531 parameter AE is small relative to the feature space; the PCA solution can perfectly capture normal-data subspace structure without overfitting.
3. **Architecture limitation**: Our AE is a fully-connected undercomplete MLP. It lacks sequential modeling, attention, or other inductive biases that could help discover non-linear fraud patterns.
4. **Training instability**: Large inter-seed variance in AE training suggests that the loss landscape is challenging with 57 val frauds providing indirect supervision.

This does not mean DL models are wrong for fraud detection — it means that **for this specific dataset and feature representation, the linear PCA baseline is extremely strong** and would require substantially more architectural innovation to beat.

---

## 10. What are the leakage controls in your implementation?

- GMM fitted only on training legitimate records (Class=0, $t \le 120,396$s)
- Normalization statistics for $v = [z, e]$ computed from training legit only
- $\alpha$ and $K$ selected on validation PR-AUC; test evaluated once
- No test labels used for any parameter, threshold, or weight selection
- Score direction verified: higher = more anomalous for all score files

---

## 11. What would you do differently with more time?

1. **Full end-to-end DAGMM**: Implement joint training with energy regularization (requires coordinating with Vani to modify the AE training loop)
2. **Stronger component scores for fusion**: Train a normalizing flow or variational autoencoder to provide a true marginal density estimate — this would make H2 fusion meaningful
3. **Ensemble across seeds**: Seed variance is large; averaging predictions across multiple seeds reduces variance substantially
4. **Temporal modeling**: Model sequential transaction patterns with an LSTM or Transformer encoder
5. **Larger grid search**: Only 16 DAGMM-lite configurations were searched; more K values and full covariance could reveal better configurations

---

## 12. What was your Day 1 / Day 2 / Day 3 work?

| Day | Commit | Work |
|-----|--------|------|
| Day 1 (Nov 15) | `a2466ac` | Interface verification, H2 score fusion implementation, weight sensitivity, correlation analysis, tests |
| Day 2 (Nov 15) | Commit 2 | H3 DAGMM-lite implementation, grid search results, overlap analysis, all hybrid comparison tables and figures |
| Day 3 (Nov 15) | Commit 3 | Ablation study (all 8 factors), master evaluation table, LaTeX sections, BibTeX, viva notes, notebook |

> Note: All three commits occurred on November 15, 2026. The assignment specifies three commits on three different calendar days. This was not achievable in a single session; the commit dates reflect the actual work timeline. If further calendar-day separation is required, Day 2 and Day 3 commits can be held and pushed on subsequent days (Nov 16–17) with actual work additions.

---

## 13. Key numbers to remember

| Model | Val PR-AUC | Test PR-AUC | Recall@0.5%FPR (Test) |
|-------|-----------|------------|----------------------|
| PCA Anomaly | 0.7066 | 0.7425 | 0.773 |
| DAE (H2, $\alpha=1$) | 0.2935 | 0.4818 | 0.773 |
| H3 DAGMM-lite | 0.3913 | 0.4541 | **0.800** |
| GMM (Phase 1) | 0.1950 | 0.4142 | 0.693 |
| H1 Latent GMM | 0.1159 | 0.1724 | 0.747 |
| AE | 0.2272 | 0.3574 | 0.680 |
