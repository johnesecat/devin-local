---
name: lint
description: How to run lint + format checks.
scope: lint, ruff, format
---

devin-local uses `ruff` for linting and formatting. From the repo root:

- Check: `python -m ruff check .`
- Format check: `python -m ruff format --check .`
- Auto-fix: `python -m ruff check . --fix` then `python -m ruff format .`

Configuration lives in `pyproject.toml` under `[tool.ruff]`.
