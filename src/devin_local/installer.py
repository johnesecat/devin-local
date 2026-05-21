"""Installer helpers shared by the CLI and the GUI.

Kept Qt-free so it can be imported by ``devin-local install`` without
requiring the ``[gui]`` extra.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: Map from inference backend name -> list of pip extras to install.
BACKEND_EXTRAS: dict[str, tuple[str, ...]] = {
    "layered": ("layered",),
    "hf": ("hf",),
}


def is_editable_install() -> bool:
    """Return True if devin-local is installed in editable (-e) mode."""
    try:
        import devin_local  # type: ignore[import-not-found]
    except ImportError:
        return False
    path = Path(getattr(devin_local, "__file__", "") or "").resolve()
    if not path.exists():
        return False
    parts = {p.lower() for p in path.parts}
    return "site-packages" not in parts and "dist-packages" not in parts


def find_source_root() -> Path | None:
    """If running editable, return the path containing ``pyproject.toml``."""
    try:
        import devin_local  # type: ignore[import-not-found]
    except ImportError:
        return None
    path = Path(getattr(devin_local, "__file__", "") or "").resolve()
    candidate = path
    for _ in range(6):
        candidate = candidate.parent
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def pip_command(extras: tuple[str, ...]) -> list[str]:
    """Build the ``pip install`` argv for installing one or more extras."""
    spec = ",".join(extras)
    if is_editable_install():
        root = find_source_root()
        if root is not None:
            return [sys.executable, "-m", "pip", "install", "-e", f"{root}[{spec}]"]
    return [sys.executable, "-m", "pip", "install", "--upgrade", f"devin-local[{spec}]"]


def backend_dep_probe(backend: str) -> tuple[bool, str]:
    """Probe whether the named backend's heavy deps import cleanly.

    Returns ``(ok, details)``. For ``ollama`` this is always (True, "ready").
    """
    if backend == "ollama":
        return (True, "ready")
    if backend == "layered":
        try:
            import airllm  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return (False, f"airllm import failed: {exc}")
        return (True, "ready")
    if backend == "hf":
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
            import transformers  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return (False, f"transformers/torch import failed: {exc}")
        return (True, "ready")
    return (False, f"unknown backend {backend!r}")


def install_extras_sync(extras: tuple[str, ...]) -> int:
    """Run ``pip install`` synchronously, streaming output to stdout.

    Returns the pip exit code. Used by the CLI ``devin-local install`` command
    and by tests.
    """
    argv = pip_command(extras)
    print("$", " ".join(argv), flush=True)
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(argv, env=env)  # noqa: S603
