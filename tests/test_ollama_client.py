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


def test_close_aborts_blocked_chat_stream_within_one_second():
    """Calling ``close()`` from another thread while ``chat_stream`` is
    blocked waiting on the first NDJSON line from a slow server must
    unwind in well under a second. Without the socket-shutdown abort,
    the blocked read would wait for the full HTTP read-timeout (600s
    by default) — which is exactly the live-test failure mode where
    the Stop button click did nothing for several minutes.
    """
    import contextlib
    import socket as _socket
    import threading
    import time

    stop_event = threading.Event()
    ready = threading.Event()
    ready_port: list[int | None] = [None]

    def _slow_server() -> None:
        srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(0.25)
        ready_port[0] = srv.getsockname()[1]
        ready.set()
        conns: list[_socket.socket] = []
        try:
            while not stop_event.is_set():
                try:
                    conn, _addr = srv.accept()
                except TimeoutError:
                    continue
                conns.append(conn)
                # Send headers but never the body so iter_lines() blocks.
                conn.send(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
        finally:
            for c in conns:
                with contextlib.suppress(Exception):
                    c.close()
            srv.close()

    server_thread = threading.Thread(target=_slow_server, daemon=True)
    server_thread.start()
    assert ready.wait(timeout=2.0), "slow test server never started"
    port = ready_port[0]
    assert port is not None

    client = OllamaClient(host=f"http://127.0.0.1:{port}", timeout=30.0)
    result: list[tuple[str, str]] = []

    def _consume() -> None:
        try:
            for _ in client.chat_stream(
                model="dummy",
                messages=[ChatMessage(role="user", content="hi")],
            ):
                pass
        except OllamaError as exc:
            result.append(("ollama_error", str(exc)))
        except Exception as exc:  # noqa: BLE001
            result.append((type(exc).__name__, str(exc)))
        result.append(("done", ""))

    consumer = threading.Thread(target=_consume, daemon=True)
    consumer.start()
    # Wait until the request is in-flight (server has the connection).
    time.sleep(0.8)
    t_close = time.monotonic()
    client.close()
    consumer.join(timeout=3.0)
    elapsed = time.monotonic() - t_close
    stop_event.set()

    assert not consumer.is_alive(), (
        f"chat_stream consumer still blocked {elapsed:.2f}s after close(); "
        f"Stop button would feel broken in the GUI"
    )
    assert elapsed < 1.0, (
        f"close() took {elapsed:.2f}s to unblock chat_stream "
        f"(expected <1s); GUI Stop button latency would be too high"
    )
    assert result, "consumer produced no result"
    # First entry should be the abort error; the "done" sentinel follows.
    kind = result[0][0]
    assert kind in {"ollama_error"}, f"unexpected exit kind {kind!r}: {result}"
