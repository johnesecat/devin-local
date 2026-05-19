"""`InferenceBackend` adapter for the Ollama HTTP API.

Wraps `devin_local.ollama_client.OllamaClient` so it satisfies the
`InferenceBackend` contract. Adds real streaming support: the underlying
client now exposes a `chat_stream()` generator that yields NDJSON deltas
straight from `/api/chat?stream=true`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from devin_local.inference.backend import (
    BackendUnavailableError,
    ChatChunk,
    ChatResponse,
    InferenceBackend,
)
from devin_local.inference.types import ChatMessage
from devin_local.ollama_client import OllamaClient, OllamaError


class OllamaBackend(InferenceBackend):
    """Backend that delegates everything to a local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 600.0,
        client: OllamaClient | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._client = client or OllamaClient(host=self.host, timeout=timeout)

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> ChatResponse:
        try:
            ollama_resp = self._client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options=options,
                keep_alive=keep_alive,
            )
        except OllamaError as exc:
            raise BackendUnavailableError(f"ollama backend: {exc}") from exc
        return ChatResponse(
            message=ollama_resp.message,
            raw=ollama_resp.raw,
            eval_count=ollama_resp.eval_count,
            prompt_eval_count=ollama_resp.prompt_eval_count,
            total_duration_ns=ollama_resp.total_duration_ns,
            done=ollama_resp.done,
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> Iterator[ChatChunk]:
        try:
            for chunk in self._client.chat_stream(
                model=model,
                messages=messages,
                tools=tools,
                options=options,
                keep_alive=keep_alive,
            ):
                yield ChatChunk(
                    delta=chunk.delta,
                    done=chunk.done,
                    message=chunk.message,
                    raw=chunk.raw,
                )
        except OllamaError as exc:
            raise BackendUnavailableError(f"ollama backend: {exc}") from exc

    def summarize(
        self,
        model: str,
        text: str,
        max_tokens: int = 512,
        instruction: str | None = None,
    ) -> str:
        try:
            return self._client.summarize(
                model=model,
                text=text,
                max_tokens=max_tokens,
                instruction=instruction,
            )
        except OllamaError as exc:
            raise BackendUnavailableError(f"ollama backend: {exc}") from exc

    def is_available(self) -> bool:
        return self._client.is_available()

    def list_models(self) -> list[str]:
        try:
            return self._client.list_models()
        except OllamaError as exc:
            raise BackendUnavailableError(f"ollama backend: {exc}") from exc

    def close(self) -> None:
        self._client.close()
