---
name: run-tests
description: How to run the devin-local test suite.
scope: testing, pytest, ci
---

From the repository root, run the test suite with:

```
python -m pytest -q
```

If you are adding a new test, place it under `tests/` and name the file
`test_<thing>.py`. Use `pytest.mark.asyncio` for async tests (the project
uses `asyncio_mode = "auto"` in `pyproject.toml`, so a plain `async def
test_*` function works too).
