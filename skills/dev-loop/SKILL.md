---
name: dev-loop
description: Full development lifecycle — from theory to GitHub release. Teams Claude with GitHub Copilot across all stages.
---

# Dev Loop Skill

## Use This Skill When

1. Starting a new feature or research direction and need to move it from idea to shipped.
2. Returning to the project after a gap and need to find the right entry point in the cycle.
3. Preparing to merge and release a completed feature branch.
4. Creating GitHub issues from TODO/IMPLEMENTED gaps or surfaced ideas.

## Lifecycle States

```
THEORY → ISSUES → IMPLEMENTING → PR_OPEN → REVIEWING → MERGED → TRAINING → RELEASE
```

Detection: run `git status` + `gh pr list` + `gh issue list` to locate current state.

---

## Stage Guide

### THEORY
**Entry:** No branch, just ideas or TODO items.

1. Read `IMPLEMENTED.md` (what was done) and `TODO.md` (what's next).
2. Read recent memory files for context on current research direction.
3. Formulate a concrete hypothesis: what change, what expected outcome, how to measure.
4. Write it as a GitHub issue (next stage).

### ISSUES
**Entry:** Have a hypothesis or TODO backlog.

```bash
gh issue create --title "..." --label "enhancement,research" --body "..."
```

Label conventions:
- `enhancement` — new feature or improvement
- `research` — experiment or hypothesis
- `evolution` — evolutionary algorithm / coevolution related
- `environment` — world simulation
- `metrics` — measurement and observability
- `infrastructure` — CI, tooling, developer experience

**GH Copilot teaming:** Use Copilot Workspace on GitHub.com to scaffold issue-to-branch when the change is well-scoped; Claude handles open-ended research implementation.

### IMPLEMENTING
**Entry:** Issue(s) created, ready to code.

```bash
git checkout -b feat/<short-name>
# ... implement, commit regularly
git commit -m "feat(...): description\n\nCloses #N"
git push origin feat/<short-name>
```

Rules:
- Commit at each logical unit (don't squash until PR).
- Reference `Closes #N` in commit messages so issues auto-close on merge.
- Run smoke tests before each push: `python -m ai_embedded_dynamic_diversity.train.cli run --epochs 1 --device cpu --coevolution --population-size 2`.

### CI GOTCHAS

**Single-command Typer apps:** Use the registered entry point (`uv run add-train`), not `python -m ... cli run`. Single-command Typer apps do not accept the command name as an argument — `run` becomes "unexpected extra argument". Check `pyproject.toml [project.scripts]` for the correct entry point names.

**Parallel agent worktrees:** Agents cannot execute git/shell commands in CI-isolated sessions. They write files but cannot push. Always check `git worktree list` after agents complete; force-remove stale ones with `git worktree remove -f -f`. If the branch name is locked to a worktree, force-remove first then recreate the branch.

**Pre-merge CI check:** Before opening a PR, verify the workflow command locally:
```bash
uv run add-train --epochs 1 --batch-size 2 --unroll-steps 3 --device cpu --no-strict-device --coevolution --population-size 2 --save-path /tmp/smoke.pt
uv run add-sim profiler --embodiment hexapod --steps 5 --batch-size 2 --device cpu --output /tmp/smoke-profile.json
```

### PR_OPEN
**Entry:** Feature branch ready for review.

```bash
gh pr create --title "feat(...): ..." --body "$(cat <<'EOF'
## Summary
...

## Checklist
- [ ] Documentation updated (IMPLEMENTED.md, README.md, TODO.md)
- [ ] Issues linked (Closes #N, #M)
- [ ] Artifacts produced
- [ ] Smoke test passing
EOF
)"
```

**GH Copilot teaming:** Open the PR on GitHub.com → use Copilot Chat on the diff for an independent second-opinion review. Copilot focuses on code correctness; Claude focuses on research goal alignment.

### REVIEWING
**Entry:** PR is open, review comments exist.

```bash
# Copilot's inline comments don't show in gh pr view --comments.
# Fetch them explicitly:
gh api "repos/displague/ai-embedded-dynamic-diversity/pulls/<N>/comments" \
  | python -c "import sys,json; [print(f'[{c[\"path\"]}:{c.get(\"line\",\"?\")}] {c[\"body\"][:200]}') for c in json.load(sys.stdin)]"
```

**Decision tree for each comment:**

| Comment type | Action |
|---|---|
| Bug / correctness | Fix inline, commit, push, reply with commit SHA |
| Unused import/param | Remove immediately (1-line fix) |
| Hardcoded value / configurability | Add env var or CLI flag with sensible default |
| Dead/contradictory code/docs | Remove or rewrite |
| Scope creep / enhancement | `gh issue create`, reply "tracked as #M" |
| Style nit | Fix if trivial (< 5 min); dismiss with one-sentence rationale if not worth it |
| Regression concern | Run targeted eval, post result as reply |

**After all comments resolved:** Reply on the PR with a single comment listing every issue number and its fix commit. Then merge.

**Note on `--frozen` flag for `uv sync` in CI:** Always use `uv sync --frozen` in GitHub Actions. Without `--frozen`, `uv` will update dependencies if the lockfile drifts from `pyproject.toml`, silently breaking reproducibility.

**Note on README/workflow consistency:** If the workflow trigger changes, update the README CI section to match. Reviewers catch this mismatch.

### MERGED
**Entry:** All blockers resolved.

```bash
gh pr merge <N> --squash --delete-branch
git checkout master && git pull origin master
```

Squash merge preserves a clean master history with all context in the commit body.

### POST-MERGE CLEANUP (do immediately after each merge)

```bash
# 1. Remove the feature branch locally if it still exists
git branch -d feat/<name> 2>/dev/null || true

# 2. Remove any lingering agent worktrees from this feature
git worktree list          # find any .claude/worktrees/agent-* entries
git worktree remove -f -f ".claude/worktrees/agent-<id>"  # for each one
git branch -D worktree-agent-<id>                          # remove worktree branch

# 3. Read any Copilot review comments that arrived AFTER merge
gh pr view <N> --comments | grep -A 10 "copilot"
# For each actionable comment: commit fix to master, comment on PR pointing to fix commit

# 4. If CI failed AFTER merge (on master), fix immediately
gh run list --branch master --limit 3
# Fix → commit to master → push

# 5. Update TODO.md: mark the issue(s) closed
sed -i 's/^- \[ \] <issue text>/- [x] <issue text> (Closes #N)/' TODO.md
git add TODO.md && git commit -m "docs(todo): mark #N closed"
```

**Copilot review timing:** Copilot reviews often arrive after merge. Always check `gh pr view <N> --comments` even on merged PRs. If actionable: fix on master, commit with message referencing the PR, comment on the PR with the commit SHA.

### TRAINING
**Entry:** Feature merged; need to measure whether new environment/loss/coupling actually improves champions.

```bash
python scripts/eval_evolve_new_env.py
# or for general purpose:
python -m ai_embedded_dynamic_diversity.train.parallel_cli \
  --variants 4 --profile pi5 --epochs 40 --device cuda \
  --coevolution --population-size 6 \
  --init-weights-cycle "artifacts/model-core-champion-v09.pt,artifacts/model-core-champion-v08.pt" \
  --out-dir artifacts/parallel-new-feature
```

Improvement threshold: new champion must exceed prior best on `new_env_v1` fitness OR improve GDI while maintaining transfer.

### RELEASE
**Entry:** Training complete, improvement confirmed.

**Step 1:** Documentation
```bash
# Append to IMPLEMENTED.md
# Update README.md (new flags, profiles, examples)
# Mark TODO.md items complete; add new items
git add IMPLEMENTED.md README.md TODO.md
git commit -m "docs: update for vX.Y.Z release"
```

**Step 2:** Visualizations
```bash
python scripts/viz_new_env.py   # or equivalent
cp artifacts/.../viz-*.gif docs/assets/
cp artifacts/.../viz-*.png docs/assets/
git add docs/assets/
git commit -m "docs(assets): add vX.Y.Z release visualizations"
git push origin master
```

**Step 3:** GitHub issues for out-of-scope work (before release)
```bash
gh issue create --title "..." --label "..."  # 1 per surfaced idea
```

**Step 4:** Release
```bash
gh release create vX.Y.Z \
  --title "vX.Y.Z — Short description" \
  --notes-file /tmp/release-notes.md \
  --target master \
  "artifacts/.../champion.pt#champion.pt" \
  "artifacts/.../champion.metrics.json#champion.metrics.json" \
  "artifacts/.../phase-eval.json#phase-eval.json" \
  "artifacts/.../viz-*.gif#viz-*.gif"
```

Release notes structure: summary → what changed (numbered, with embedded GIFs) → champion benchmark table → training trajectory PNG → hazard/scenario GIFs → attached artifacts table → issues opened.

---

## GH Copilot Teaming

| Stage | Claude | GitHub Copilot |
|---|---|---|
| THEORY | Research synthesis, hypothesis formulation | — |
| ISSUES | Write issue bodies, create labels | Suggest issue templates |
| IMPLEMENTING | Full implementation, training scripts | Code review hints, autocompletion |
| PR_OPEN | Write PR body, link issues | Independent diff review (Copilot Chat) |
| REVIEWING | Address comments, open new issues | Identify correctness bugs in diff |
| MERGED | Squash merge | — |
| TRAINING | Run eval/evolve scripts, interpret results | — |
| RELEASE | Write release notes, embed GIFs, upload artifacts | — |

---

## Reference Loading

- `references/release-history.md`: past release summaries and version rationale.
- `TODO.md` (repo root): current backlog with completion status.
- `IMPLEMENTED.md` (repo root): full feature chronology.
