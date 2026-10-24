# Contributing — Team T033, Project 26

## Branch Convention

| Member | Branch |
|---|---|
| Kushal Sarkar | `feat/lead` |
| Prachee Dhar | `feat/prachee` |
| Apoorva Choudhary | `feat/apoorva` |
| Vani Rudra | `feat/vani` |
| Krishna Pramod Dubale | `feat/krishna` |

Each member works exclusively on their own branch. Open a PR to `main` when ready. Kushal (lead) reviews and merges.

## Commit Prefix Convention

Every commit message must start with the member's prefix:

```
[lead]     Kushal
[prachee]  Prachee
[apoorva]  Apoorva
[vani]     Vani
[krishna]  Krishna
```

Example: `[apoorva] add GMM AIC/BIC grid search`

## Workflow

```bash
git pull origin main                     # always start from latest main
git checkout -b feat/<your-name>         # or git checkout feat/<your-name>

# … make changes, run tests …

git add <your-files>
git commit -m "[yourname] describe what you did"
git fetch origin && git rebase origin/main   # stay current with main
git push -u origin feat/<your-name>
# Then open a Pull Request on GitHub
```

## Rules

1. **Never commit `data/raw/creditcard.csv`** or any other large/raw data file.
2. **Never commit model files > 100 MB** to the repository.
3. **Never force-push** (`git push --force`) to any branch. If rebase creates
   conflicts, resolve them locally and push normally.
4. **Commit often** — individual commit history is evidence of contribution
   for the viva.  Each commit must represent a real, runnable increment.
5. **Edit only the files listed under "Files you own"** in your prompt file.
   To request a change in a shared file, write it to `docs/requests/<name>.md`
   and notify Kushal.
6. **Commit dates must be genuine** — never set or alter commit dates
   (no `GIT_AUTHOR_DATE`, `GIT_COMMITTER_DATE`, `--date` flags, or history
   rewriting).  Commits carry the true time they were made.
7. **Rebase on `origin/main` before pushing** to avoid merge conflicts.
8. **Each member owns distinct files** — PRs must not edit another member's
   files to prevent conflicts.
9. **Do not copy another team's code.**  Shared ideas from the problem statement
   are fine; identical implementation is not.
10. **Do not fabricate commits, numbers, or results.**  Vivas are individual.

## PR Checklist

Before opening a PR:
- [ ] `pytest tests/ -v` passes (or failures are documented with a known cause)
- [ ] Notebook cells are executed with real outputs
- [ ] No data files committed (`git status` shows no CSVs/npy in staging)
- [ ] Commit messages follow the prefix convention
- [ ] `docs/viva_notes/<name>.md` is updated or complete
