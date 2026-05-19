"""Ollama HTTP client with native tool calling support.

This module intentionally uses raw `httpx` rather than the `ollama` package so
that we have no extra dependency and can adapt easily to API drift. The Ollama
chat API is documented here: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from devin_local.inference.types import ChatMessage

DEFAULT_HOST = "http://127.0.0.1:11434"

# Re-exported so existing imports `from devin_local.ollama_client import ChatMessage` keep working.
__all_message__ = ChatMessage  # noqa: F841  (suppress unused; kept for clarity)


@dataclass
class ChatResponse:
    """A normalized response from Ollama's `/api/chat` endpoint."""

    message: ChatMessage
    raw: dict[str, Any]
    eval_count: int = 0
    prompt_eval_count: int = 0
    total_duration_ns: int = 0
    done: bool = True


@dataclass
class StreamChunk:
    """One incremental piece of a streamed `/api/chat?stream=true` response."""

    delta: str = ""
    done: bool = False
    message: ChatMessage | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaError(RuntimeError):
    """Raised when Ollama returns an error response or is unreachable."""


class OllamaClient:
    """Minimal Ollama client tailored for tool-calling agents.

    The client exposes `chat()` (single-shot, returns a ChatResponse) and
    `is_available()` (probes the daemon). All network I/O is synchronous —
    the agent loop is single-threaded by design.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---------- public API ----------

    def is_available(self) -> bool:
        """Return True if the Ollama daemon answers within the configured timeout."""
        try:
            resp = self._client.get(f"{self.host}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        """Return the names of locally installed Ollama models."""
        try:
            resp = self._client.get(f"{self.host}/api/tags", timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        format: str | None = None,
        keep_alive: str | int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama (non-streaming)."""
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options
        if format:
            body["format"] = format
        if keep_alive is not None:
            body["keep_alive"] = keep_alive

        try:
            resp = self._client.post(f"{self.host}/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        msg_data = data.get("message", {}) or {}
        message = ChatMessage(
            role=msg_data.get("role", "assistant"),
            content=msg_data.get("content", "") or "",
            tool_calls=list(msg_data.get("tool_calls", []) or []),
        )
        return ChatResponse(
            message=message,
            raw=data,
            eval_count=int(data.get("eval_count", 0) or 0),
            prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
            total_duration_ns=int(data.get("total_duration", 0) or 0),
            done=bool(data.get("done", True)),
        )

    def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        format: str | None = None,
        keep_alive: str | int | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion from Ollama as NDJSON.

        Yields one `StreamChunk` per NDJSON record. The final chunk has
        `done=True` and carries the accumulated `ChatMessage` (with structured
        `tool_calls` if the model emitted any).
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options
        if format:
            body["format"] = format
        if keep_alive is not None:
            body["keep_alive"] = keep_alive

        accumulated_content: list[str] = []
        accumulated_tool_calls: list[dict[str, Any]] = []

        try:
            with self._client.stream(
                "POST", f"{self.host}/api/chat", json=body, timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    body_text = resp.read().decode("utf-8", errors="replace")[:500]
                    raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {body_text}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg_data = record.get("message", {}) or {}
                    delta = msg_data.get("content", "") or ""
                    if delta:
                        accumulated_content.append(delta)
                    chunk_tool_calls = msg_data.get("tool_calls") or []
                    if chunk_tool_calls:
                        accumulated_tool_calls.extend(chunk_tool_calls)
                    done = bool(record.get("done", False))
                    if done:
                        final = ChatMessage(
                            role="assistant",
                            content="".join(accumulated_content),
                            tool_calls=accumulated_tool_calls,
                        )
                        yield StreamChunk(delta=delta, done=True, message=final, raw=record)
                    else:
                        yield StreamChunk(delta=delta, done=False, raw=record)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama stream failed: {exc}") from exc

    def summarize(
        self,
        model: str,
        text: str,
        max_tokens: int = 512,
        instruction: str | None = None,
    ) -> str:
        """Ask the model to summarize a chunk of text (for context compression)."""
        prompt = instruction or (
            "Summarize the following transcript into a dense, faithful note. "
            "Preserve all decisions, file paths, identifiers, and pending actions. "
            "Omit chit-chat. Use compact prose, not bullet points."
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=text),
        ]
        response = self.chat(
            model=model,
            messages=messages,
            options={"num_predict": max_tokens, "temperature": 0.2},
        )
        return response.message.content.strip()


def parse_tool_call_arguments(raw: Any) -> dict[str, Any]:
    """Normalize the `arguments` field of an Ollama tool call into a dict.

    Some Ollama backends return arguments as a JSON-encoded string instead of
    a JSON object. This helper handles both shapes.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}
        if isinstance(parsed, dict):
            return parsed
        return {"_value": parsed}
    return {}
