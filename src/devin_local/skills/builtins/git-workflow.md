---
name: git-workflow
description: Conventional git workflow for this repo.
scope: git, commit, branch, pr
---

- Branch naming: `feature/<short-description>` or `fix/<short-description>`.
- Always read the diff with `git diff` before committing.
- Write commit messages in the imperative ("Add X", not "Added X").
- One logical change per commit when feasible.
- Don't force-push to `main`.
