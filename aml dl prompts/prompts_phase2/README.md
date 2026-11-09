# Phase 2 prompts (Deep Learning + Hybrid) — Team T033

Use these prompts only after Phase 1 is completed, reviewed, merged and tagged `phase1-final`. They are retained for future work and are **not part of the current submission**.

Keep this folder at `prompts_phase2/`. Do not copy or move it into `docs/prompts/phase2/`, and do not create a duplicate prompts directory. When Phase 2 begins, follow the task order and dependencies in the relevant prompt files. Commit genuine, meaningful work; never alter commit dates.

## MANDATORY FIRST STEP — READ THE ENTIRE EXISTING PHASE 2 DOCUMENTATION

Before performing any task, you MUST thoroughly inspect and read the existing Phase 2-related Markdown files in this repository.

Do not start implementation, create documentation, or modify files until this reading and understanding step is complete.

### 1. Read all Phase 2 prompt files

Read every existing Markdown file inside:

- `prompts_phase2/`
- Any other directory containing Phase 2-specific prompts or instructions.

This includes:
- `P2_01_Krishna_DL_Literature_Pipeline.md`
- `P2_02_Apoorva_LatentGMM_Theory.md`
- `P2_03_Vani_Autoencoder.md`
- `P2_04_Prachee_Hybrid_Ablation.md`
- `P2_05_Kushal_FINAL.md`
- `prompts_phase2/README.md`

Read all five prompts, not just your assigned prompt. Understand everyone's responsibilities, dependencies, deliverables and integration requirements.

### 2. Read all other Phase 2-related Markdown files

Search the entire repository for additional Phase 2-related `.md` files, including files under:

- `docs/`
- `guides/`
- `reports/`
- Other relevant project directories.

Read every relevant Phase 2 document, including existing architecture notes, research notes, implementation plans, experiment documentation and team instructions.

Do not assume that a file is irrelevant simply because it is outside `prompts_phase2/`.

### 3. Understand the existing Phase 1 implementation

Read the relevant Phase 1 documentation and inspect the existing implementation to understand:

- Dataset and preprocessing.
- Train, validation and test splits.
- Feature engineering.
- GMM implementation.
- Classical baselines.
- Evaluation metrics.
- Score conventions.
- Saved artifacts and results.

Phase 2 must extend Phase 1 without breaking its existing work.

### 4. Build an understanding before proceeding

Before starting your assigned task, provide a short internal audit in your response containing:

1. Which Markdown files you read.
2. What the existing Phase 2 plan is.
3. What the other four members are responsible for.
4. How your assigned task connects to their work.
5. Existing implementation or documentation that can be reused.
6. Any conflicts, missing information or inconsistencies you discovered.

### 5. Important rules

- Do not skip files because they appear repetitive.
- Do not assume that the current prompt contains the entire project context.
- Do not duplicate work already completed by another member.
- Do not overwrite existing documentation without checking it first.
- Do not silently resolve conflicting instructions; report them.
- Distinguish planned work from work that has actually been implemented and verified.
- Preserve existing project structure and user modifications.

Only after completing this review should you proceed with your assigned task.

# Phase 2 — Deep Learning & Hybrid Anomaly Detection

**Project:** Project 26 — Card Fraud Triage  
**Team:** T033  
**Phase:** 2 — Deep Learning

## 1. Objective

Phase 1 established classical anomaly-detection baselines, including GMM, PCA, Isolation Forest, LOF and One-Class SVM.

Phase 2 extends this work by investigating:

1. Deep Autoencoder anomaly detection.
2. Latent-space Gaussian Mixture Model.
3. Hybrid Autoencoder + latent-GMM scoring.

The central research question is:

**Does learning a compressed representation using a Deep Autoencoder and modelling its latent distribution using GMM improve fraud detection compared with classical anomaly-detection methods?**

## 2. Team assignments

| Prompt | Member | Responsibility |
|---|---|---|
| P2_01 | Krishna | Deep Learning literature and pipeline design |
| P2_02 | Apoorva | Latent-space GMM theory and design |
| P2_03 | Vani | Deep Autoencoder implementation |
| P2_04 | Prachee | Hybrid model and ablation study |
| P2_05 | Kushal | Final integration and report |

## 3. Execution order

### Step 1 — Parallel assignments

Prachee, Apoorva, Vani and Krishna can begin their respective assignments in parallel.

Each member must first inspect the existing repository and understand Phase 1.

### Step 2 — Interface coordination

Before integration:

- Vani and Apoorva must agree on the encoder and latent-vector interface.
- Vani and Krishna must agree on the Autoencoder score format.
- Apoorva and Krishna must agree on the latent-GMM score format.
- All members must use the existing data split and compatible evaluation conventions.

### Step 3 — Individual handoff

Each member should provide:

- Files created or modified.
- Implementation or documentation summary.
- Actual results, if experiments were performed.
- Commands executed.
- Tests and verification results.
- Remaining issues.

### Step 4 — Final integration

Kushal runs P2_05 after the other assignments are available.

The integration owner audits all deliverables before making changes.

## 4. Important experimental rules

1. Train the Autoencoder using legitimate training transactions only.
2. Fit latent GMM using legitimate training latent vectors only.
3. Use validation data for model selection and hybrid weight selection.
4. Keep the test set untouched until final evaluation.
5. Preserve the chronological data split from Phase 1.
6. Use consistent transaction ordering across all score arrays.
7. Ensure higher anomaly scores always indicate greater abnormality.
8. Never fabricate or manually invent results.
9. Preserve all Phase 1 models, results and artifacts.

## 5. Evaluation metrics

Use the existing Phase 1 evaluation protocol wherever possible.

Required metrics:

- PR-AUC.
- Recall at 0.1%, 0.5% and 1.0% FPR.
- Precision at review capacities.
- Runtime where available.
- Relevant uncertainty analysis.

Accuracy must not be used as the primary evaluation metric because the dataset is extremely imbalanced.

## 6. Expected Phase 2 deliverables

- Literature review and methodology.
- Autoencoder implementation and saved model.
- Latent-space GMM implementation or agreed integration.
- Hybrid anomaly scoring.
- Ablation study.
- Phase 1 vs Phase 2 comparison.
- Updated report and visualizations.
- Reproducible commands and tests.

## 7. Repository safety

- Do not delete existing Phase 1 files.
- Do not overwrite another member's work.
- Do not change shared evaluation interfaces without coordination.
- Do not add unnecessary dependencies.
- Do not commit or push without approval.
- Keep raw datasets and generated artifacts out of Git when they are covered by `.gitignore`.

## 8. How to use these prompts

1. Open the repository in Antigravity.
2. Open your assigned `P2_XX` Markdown file.
3. Give Antigravity the complete prompt as its task instructions.
4. Let it inspect the repository before implementing.
5. Review its final handoff and verify that files and experiments actually exist.
6. Share the completion summary with the team.

**Do not run P2_05 until P2_01 through P2_04 have completed their work.**

## MANDATORY GIT WORKFLOW — CREATE A NEW BRANCH BEFORE WORK

After completing the mandatory documentation review, but BEFORE modifying or creating any project files, prepare an isolated Git branch for your assignment.

### 1. Inspect Git status

Run:

```bash
git status
git branch --show-current
git remote -v
```

Check:
- Current branch.
- Uncommitted changes.
- Existing branch names.
- Remote configuration.

IMPORTANT:
- Never discard existing user changes.
- Never run `git reset --hard`.
- Never use `git clean -fd`.
- Never overwrite or delete someone else's work.
- Do not stash, commit or discard pre-existing changes without approval.
- If the working tree contains unrelated uncommitted changes that could be affected by switching branches, stop and report the situation before proceeding.

### 2. Synchronize with the correct base branch

Identify the team's agreed integration/base branch.

Fetch the latest remote references:

```bash
git fetch origin
```

Do not assume that `main` is the correct base branch. Inspect the available branches and use the team's agreed base branch.

If the base branch has uncommitted changes or the local branch is behind the remote, do not overwrite anything. Report the issue and request clarification if necessary.

### 3. Create your dedicated branch

Use the branch name assigned to you in this prompt.

If the branch does not already exist:

```bash
git switch -c YOUR_ASSIGNED_BRANCH YOUR_BASE_BRANCH
```

If it already exists, inspect it and switch to it only after confirming it is the correct branch and there are no unsafe working-tree changes.

Verify:

```bash
git branch --show-current
git status
```

### 4. Work only on your assigned branch

All files you create or modify for your assignment must be developed on your dedicated branch.

Do not:
- Work directly on the main or integration branch.
- Switch to another member's feature branch.
- Merge other members' branches.
- Push directly to the main branch.
- Modify unrelated files.

### 5. Before completing your assignment

Provide:
- Branch name.
- Base branch used.
- Files created.
- Files modified.
- Git status.
- Tests and checks performed.

Do not commit or push unless explicitly authorized by the team lead.

The purpose of this workflow is to keep every member's implementation isolated and make final integration safe.