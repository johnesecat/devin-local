"""`InferenceBackend` abstract base + shared response types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from devin_local.inference.types import ChatMessage


class BackendUnavailableError(RuntimeError):
    """Raised when a backend is selected but its dependencies / runtime are missing.

    For example: choosing `--backend layered` without `pip install devin-local[layered]`,
    or choosing `--backend ollama` when the daemon is not running.
    """


@dataclass
class ChatResponse:
    """A normalized completion response from any backend."""

    message: ChatMessage
    raw: dict[str, Any] = field(default_factory=dict)
    eval_count: int = 0
    prompt_eval_count: int = 0
    total_duration_ns: int = 0
    done: bool = True


@dataclass
class ChatChunk:
    """One incremental piece of a streamed response.

    A stream yields zero or more chunks with `delta` text, followed by exactly
    one final chunk with `done=True` and the full normalized `ChatMessage`
    (including any structured `tool_calls`).
    """

    delta: str = ""
    done: bool = False
    message: ChatMessage | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class InferenceBackend(ABC):
    """The contract every backend implements."""

    #: Human-readable backend identifier ("ollama", "layered", "hf").
    name: str = ""

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> ChatResponse:
        """Run a non-streaming completion. Blocks until the model is done."""

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> Iterator[ChatChunk]:
        """Run a streaming completion. Yields token deltas, then a final
        chunk with the full normalized message.

        Backends that cannot natively stream MAY emit a single chunk with
        `done=True` after running `chat()` internally. This default
        implementation does exactly that.
        """
        resp = self.chat(model, messages, tools=tools, options=options, keep_alive=keep_alive)
        yield ChatChunk(
            delta=resp.message.content,
            done=True,
            message=resp.message,
            raw=resp.raw,
        )

    @abstractmethod
    def summarize(
        self,
        model: str,
        text: str,
        max_tokens: int = 512,
        instruction: str | None = None,
    ) -> str:
        """Compress `text` into a dense summary (used by the context manager)."""

    @abstractmethod
    def is_available(self) -> bool:
        """True if this backend can actually run right now."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return the names of models this backend can serve immediately."""

    def close(self) -> None:  # noqa: B027 - intentional no-op default
        """Release any resources (HTTP sessions, model weights, etc.).

        Subclasses override when they hold state; the default is a no-op so
        every backend can safely be used as a context manager.
        """

    def __enter__(self) -> InferenceBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
