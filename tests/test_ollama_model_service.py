"""Tests for the Ollama model service (list installed, pull progress)."""

from __future__ import annotations

import httpx
import pytest
import respx

from devin_local.gui.ollama_model_service import (
    POPULAR_MODELS,
    OllamaModelService,
    PullProgress,
    _progress_from_event,
)


@pytest.fixture
def service() -> OllamaModelService:
    return OllamaModelService(host="http://127.0.0.1:11434")


@respx.mock
def test_is_available_returns_true_on_200(service: OllamaModelService) -> None:
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )
    assert service.is_available() is True


@respx.mock
def test_is_available_returns_false_on_network_error(service: OllamaModelService) -> None:
    respx.get("http://127.0.0.1:11434/api/tags").mock(side_effect=httpx.ConnectError("no daemon"))
    assert service.is_available() is False


@respx.mock
def test_list_installed_parses_response(service: OllamaModelService) -> None:
    payload = {
        "models": [
            {
                "name": "llama3.1:8b",
                "modified_at": "2025-01-01T00:00:00Z",
                "size": 4_700_000_000,
                "details": {
                    "family": "llama",
                    "parameter_size": "8.0B",
                    "quantization_level": "Q4_K_M",
                },
            },
            {"name": "qwen2.5:7b", "size": 4_400_000_000, "details": {}},
            {"model": "without-name-key", "size": 0},
        ]
    }
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json=payload)
    )
    models = service.list_installed()
    names = [m.name for m in models]
    assert "llama3.1:8b" in names
    assert "qwen2.5:7b" in names
    assert "without-name-key" in names  # falls back to "model" key
    llama = next(m for m in models if m.name == "llama3.1:8b")
    assert llama.family == "llama"
    assert llama.quantization == "Q4_K_M"
    assert llama.size_bytes == 4_700_000_000
    assert "4.4 GB" in llama.size_human or "4.5 GB" in llama.size_human


def test_list_library_returns_curated_set(service: OllamaModelService) -> None:
    library = service.list_library()
    assert library == list(POPULAR_MODELS)
    assert any(m.name == "llama3.1:8b" for m in library)


def test_progress_from_event_handles_downloading() -> None:
    evt = {"status": "downloading", "completed": 50, "total": 100, "digest": "sha256:abc"}
    progress = _progress_from_event(evt)
    assert progress.status == "downloading"
    assert progress.fraction == 0.5
    assert progress.completed == 50
    assert progress.done is False


def test_progress_from_event_handles_success() -> None:
    progress = _progress_from_event({"status": "success"})
    assert progress.done is True


def test_progress_from_event_handles_error() -> None:
    progress = _progress_from_event({"error": "model not found"})
    assert progress.error == "model not found"
    assert progress.done is True


def test_pull_progress_fraction_is_clamped() -> None:
    p = PullProgress(total=0, completed=0)
    assert p.fraction == 0.0
    p = PullProgress(total=100, completed=200)
    assert p.fraction == 1.0
