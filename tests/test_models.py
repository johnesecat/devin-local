"""Tests for the model registry."""

from __future__ import annotations

from devin_local.models import (
    DEFAULT_CONTEXT_WINDOW,
    get_model_spec,
    suggest_abliterated,
)


def test_known_model_lookup_exact():
    spec = get_model_spec("llama3.1:8b")
    assert spec.name == "llama3.1:8b"
    assert spec.context_window == 128_000
    assert spec.supports_tools is True
    assert spec.family == "llama"


def test_unknown_model_uses_default_window():
    spec = get_model_spec("totally-made-up:42b")
    assert spec.context_window == DEFAULT_CONTEXT_WINDOW


def test_prefix_match():
    spec = get_model_spec("llama3.1:8b-instruct-q4_0")
    assert spec.context_window == 128_000


def test_suggest_abliterated_returns_only_abliterated():
    suggested = suggest_abliterated()
    assert suggested, "expected at least one abliterated model"
    assert all(s.abliterated for s in suggested)
