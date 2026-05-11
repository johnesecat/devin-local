"""Tests for the Ollama HTTP client (mocked transport)."""

from __future__ import annotations

import httpx
import pytest

from devin_local.ollama_client import (
    ChatMessage,
    OllamaClient,
    OllamaError,
    parse_tool_call_arguments,
)


def _client_with_handler(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return OllamaClient(host="http://test", client=http)


def test_chat_round_trip():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = httpx.QueryParams()  # not used; just to ensure no crash
        del body
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "tool_calls": [],
                },
                "eval_count": 5,
                "prompt_eval_count": 7,
                "done": True,
            },
        )

    client = _client_with_handler(handler)
    resp = client.chat(
        model="llama3.1:8b",
        messages=[ChatMessage(role="user", content="hello")],
    )
    assert resp.message.content == "hi"
    assert resp.eval_count == 5


def test_chat_includes_tools_in_request():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    client = _client_with_handler(handler)
    client.chat(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "x"}}],
    )
    assert "tools" in captured["payload"]
    assert captured["payload"]["tools"][0]["function"]["name"] == "x"


def test_chat_http_error_raises_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client_with_handler(handler)
    with pytest.raises(OllamaError):
        client.chat(model="m", messages=[ChatMessage(role="user", content="hi")])


def test_is_available_true():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)

    client = _client_with_handler(handler)
    assert client.is_available()


def test_parse_tool_call_arguments_handles_string_json():
    assert parse_tool_call_arguments('{"a": 1}') == {"a": 1}
    assert parse_tool_call_arguments({"a": 1}) == {"a": 1}
    assert parse_tool_call_arguments("") == {}
    assert parse_tool_call_arguments("not-json") == {"_raw": "not-json"}
