> **PHASE 2 = DEEP LEARNING + HYBRID.** Prerequisite: Phase 1 is merged and tagged `phase1-final`. Do not modify any Phase 1 file or result; build on top of them. Branch name: `feat/<name>-p2`.

# PHASE 2 TASK 05 — TEAM LEAD (FINAL RUN) — Kushal Sarkar
**Run ONLY after Prachee, Apoorva, Vani and Krishna have merged their Phase 2 PRs into `main`.**

## 🤖 Antigravity prompt (paste this)
> Read `docs/prompts/phase2/P2_05_Kushal_FINAL.md` completely. Run the integration audit and print PASS/FAIL. Fix only integration problems; never alter teammates' scientific choices — log concerns in `docs/FINAL_AUDIT_P2.md`. All report numbers are read from `results/tables/*.csv`. Never alter commit dates or rewrite history. Branch `feat/lead-final-p2`, prefix `[lead]`. At the end write `docs/viva_notes/kushal_final_p2.md`.

## Step 1 — Audit
- [ ] Phase 2 commits: each member ≥ 3 commits on ≥ 3 days (flag, don't fabricate).
- [ ] Score files: ae, ae_dae, hybrid_latentgmm, hybrid_fusion, hybrid_dagmmlite × {val,test}; lengths, NaN, direction; AE trained on legit-only (assert no fraud in training tensors); early stopping uses val-legit loss only.
- [ ] Seeds: 5-seed table exists; official scores = median seed.
- [ ] Test is used only in final tables; `grep` selection code for val-only.
- [ ] `pytest` incl. new tests; notebooks 04, 07, 08, 09, 06b run end-to-end; fresh-clone reproduction (document GPU/CPU runtime).
- [ ] Merge refs, remove `[VERIFY]` only after manual check.
## Step 2 — Final comparison (`notebooks/06b_final_comparison_phase2.ipynb`)
Headline: GMM vs AE/DAE vs H1/H2/H3 vs classical baselines; recall@FPR, precision@capacity, alerts/10k, CIs, paired differences. Explicitly answer: does the AE beat the GMM significantly? Does any hybrid beat both parents with statistical support, and why/why not? Cost/complexity (params, time). Threats to validity; disclaimer.
## Step 3 — Report `reports/ieee_report/main_phase2.tex` (IEEEtran, 6–8 pages)
Abstract · Introduction (Prachee) · DL Literature (Prachee) · Data Pipeline & Regularization (Prachee/Vani) · Architecture & Theory (Vani) · Hybrid Theory & Latent-GMM (Apoorva) · Hybrids & Ablation (Krishna) · Technical Validation (Krishna) · Results & Discussion (you) · Limitations · Conclusion · Contributions · References. Numbers via `paper_numbers_p2.tex`.
## Step 4 — Slides outline `docs/slides_phase2_outline.md` (12–15 slides; backup: AE architecture, regularization table, ablation table, overlap diagram).
## Step 5 — Polish & tag
README updated with Phase 2 results; `CONTRIBUTIONS.md` regenerated from git; tag `phase2-final`; Release with PDF.
## Definition of Done
- [ ] Audit PASS. [ ] Report PDF. [ ] Slides. [ ] Fresh clone reproduces. [ ] Tag. [ ] Mock viva (`guides/VIVA_GUIDE.md` Q17–39).
## Commit rule
Never alter commit dates or rewrite history.
