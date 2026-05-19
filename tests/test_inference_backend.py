"""Tests for the inference backend abstraction and Ollama streaming."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from devin_local.inference.backend import BackendUnavailableError, ChatChunk, InferenceBackend
from devin_local.inference.factory import (
    SUPPORTED_BACKENDS,
    build_backend,
    list_available_backends,
)
from devin_local.inference.ollama_backend import OllamaBackend
from devin_local.inference.types import ChatMessage


def test_supported_backends_contains_expected() -> None:
    assert "ollama" in SUPPORTED_BACKENDS
    assert "layered" in SUPPORTED_BACKENDS
    assert "hf" in SUPPORTED_BACKENDS


def test_build_unknown_backend_raises() -> None:
    with pytest.raises(BackendUnavailableError):
        build_backend("not-a-backend")


def test_list_available_backends_reports_ollama() -> None:
    entries = list_available_backends()
    assert len(entries) == 3
    by_name = {e["name"]: e for e in entries}
    assert set(by_name) == set(SUPPORTED_BACKENDS)
    # Each entry has stable schema.
    for entry in entries:
        assert "available" in entry
        assert "details" in entry


@respx.mock
def test_ollama_backend_chat_returns_normalized_response() -> None:
    route = respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hi"},
                "done": True,
                "eval_count": 5,
                "prompt_eval_count": 4,
                "total_duration": 1_000_000,
            },
        )
    )
    backend: InferenceBackend = OllamaBackend(host="http://127.0.0.1:11434")
    try:
        resp = backend.chat(
            model="llama3.1:8b",
            messages=[ChatMessage(role="user", content="hello")],
        )
    finally:
        backend.close()
    assert route.called
    assert resp.message.role == "assistant"
    assert resp.message.content == "hi"
    assert resp.eval_count == 5
    assert resp.done is True


@respx.mock
def test_ollama_backend_stream_yields_deltas_then_final() -> None:
    body = "\n".join(
        json.dumps(rec)
        for rec in [
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "write_file", "arguments": {"path": "x"}}}
                    ],
                },
                "done": True,
            },
        ]
    )
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(200, content=body.encode("utf-8"))
    )
    backend = OllamaBackend(host="http://127.0.0.1:11434")
    chunks: list[ChatChunk] = list(
        backend.stream(
            model="llama3.1:8b",
            messages=[ChatMessage(role="user", content="hi")],
        )
    )
    backend.close()
    # 2 incremental deltas + 1 final
    deltas = [c.delta for c in chunks if not c.done]
    assert deltas == ["Hel", "lo"]
    final = [c for c in chunks if c.done]
    assert len(final) == 1
    assert final[0].message is not None
    assert final[0].message.content == "Hello"
    assert final[0].message.tool_calls
    assert final[0].message.tool_calls[0]["function"]["name"] == "write_file"


@respx.mock
def test_ollama_backend_chat_with_keep_alive_propagates_body() -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}, "done": True},
        )

    respx.post("http://127.0.0.1:11434/api/chat").mock(side_effect=_handler)
    backend = OllamaBackend(host="http://127.0.0.1:11434")
    backend.chat(
        model="llama3.1:8b",
        messages=[ChatMessage(role="user", content="hello")],
        keep_alive="10m",
    )
    backend.close()
    assert captured["body"]["keep_alive"] == "10m"


@respx.mock
def test_ollama_backend_propagates_http_error_as_backend_unavailable() -> None:
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(500, text="boom")
    )
    backend = OllamaBackend(host="http://127.0.0.1:11434")
    with pytest.raises(BackendUnavailableError):
        backend.chat(
            model="llama3.1:8b",
            messages=[ChatMessage(role="user", content="hi")],
        )
    backend.close()
