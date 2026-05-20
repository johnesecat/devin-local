"""Tests for the Qt-free installer helpers."""

from __future__ import annotations

import sys

from devin_local.installer import (
    BACKEND_EXTRAS,
    backend_dep_probe,
    pip_command,
)


def test_ollama_backend_is_always_ready() -> None:
    ok, details = backend_dep_probe("ollama")
    assert ok is True
    assert details == "ready"


def test_unknown_backend_returns_false() -> None:
    ok, details = backend_dep_probe("totally-made-up-backend")
    assert ok is False
    assert "unknown backend" in details


def test_layered_probe_returns_install_status() -> None:
    # We can't make assumptions about whether airllm is installed in the
    # test env, but the contract is: ok is bool, details is non-empty string.
    ok, details = backend_dep_probe("layered")
    assert isinstance(ok, bool)
    assert isinstance(details, str) and details


def test_pip_command_uses_current_interpreter() -> None:
    argv = pip_command(("gui",))
    assert argv[0] == sys.executable
    assert "-m" in argv
    assert "pip" in argv
    assert "install" in argv


def test_pip_command_includes_extras_spec() -> None:
    argv = pip_command(("layered", "hf"))
    # The last arg should be the spec containing both extras.
    spec = argv[-1]
    assert "layered" in spec
    assert "hf" in spec


def test_backend_extras_map_includes_layered_and_hf() -> None:
    assert "layered" in BACKEND_EXTRAS
    assert "hf" in BACKEND_EXTRAS
    assert BACKEND_EXTRAS["layered"] == ("layered",)
    assert BACKEND_EXTRAS["hf"] == ("hf",)
