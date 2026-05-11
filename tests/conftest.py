"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the src/ layout to sys.path when tests are run without `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
