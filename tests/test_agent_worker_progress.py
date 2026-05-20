"""AgentWorker progress-signal tests.

Validates that prefill / generate / done / idle progress stages get
emitted in the right order during a streaming turn, so the GUI status
bar can render a live tok/s counter that actually moves rather than
sitting blank while the model prefills.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from devin_local.agent import Agent, AgentConfig
from devin_local.gui.agent_worker import AgentWorker
from devin_local.inference.backend import (
    ChatChunk,
    ChatResponse,
    InferenceBackend,
)
from devin_local.inference.types import ChatMessage


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _ScriptedStreamBackend(InferenceBackend):
    """Streams a fixed sequence of token deltas, then a final done chunk.

    The opening pause simulates CPU prefill (no tokens for ``prefill_delay_s``)
    so we can assert the worker emits a ``prefill`` progress stage before
    any tokens arrive.
    """

    def __init__(self, deltas: list[str], prefill_delay_s: float = 0.0) -> None:
        self._deltas = deltas
        self._prefill = prefill_delay_s

    def chat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        msg = ChatMessage(role="assistant", content="".join(self._deltas))
        return ChatResponse(message=msg, eval_count=len(self._deltas))

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatChunk]:
        if self._prefill:
            time.sleep(self._prefill)
        for d in self._deltas:
            yield ChatChunk(delta=d, done=False)
            time.sleep(0.01)
        yield ChatChunk(
            delta="",
            done=True,
            message=ChatMessage(role="assistant", content="".join(self._deltas)),
        )

    def summarize(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return []


def _drain_qt_events(seconds: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def test_worker_emits_prefill_generate_done_idle_in_order(tmp_path) -> None:
    _app()
    backend = _ScriptedStreamBackend(deltas=["hello", " ", "world"])
    config = AgentConfig(
        workspace=tmp_path,
        model="llama3.2:1b",
        backend="ollama",
        max_iterations=1,
    )
    agent = Agent(config, backend=backend)
    worker = AgentWorker(agent)

    events: list[tuple[str, dict]] = []
    worker.progress_changed.connect(lambda stage, info: events.append((stage, info)))

    # Run synchronously on the test thread — submit() pumps the stream
    # observer inline as deltas arrive, no Qt threading needed.
    worker.submit("hi")
    _drain_qt_events(0.05)

    stages = [stage for stage, _ in events]
    assert "prefill" in stages, f"prefill never emitted; saw: {stages}"
    assert "generate" in stages, f"generate never emitted; saw: {stages}"
    assert "done" in stages, f"done never emitted; saw: {stages}"
    assert "idle" in stages, f"idle never emitted; saw: {stages}"
    # prefill must come before generate, generate before done, done before idle.
    p = stages.index("prefill")
    g = stages.index("generate")
    d = stages.index("done")
    i = stages.index("idle")
    assert p < g < d < i, f"stages out of order: {stages}"
    done_info = next(info for stage, info in events if stage == "done")
    assert done_info["completion_tokens"] == 3
    assert done_info["elapsed_s"] > 0


def test_worker_generate_info_includes_tok_s_and_tokens(tmp_path) -> None:
    _app()
    backend = _ScriptedStreamBackend(deltas=["a", "b", "c", "d", "e"])
    config = AgentConfig(
        workspace=tmp_path,
        model="llama3.2:1b",
        backend="ollama",
        max_iterations=1,
    )
    agent = Agent(config, backend=backend)
    worker = AgentWorker(agent)
    seen: list[tuple[str, dict]] = []
    worker.progress_changed.connect(lambda stage, info: seen.append((stage, info)))

    worker.submit("hi")
    _drain_qt_events(0.05)

    generates = [info for stage, info in seen if stage == "generate"]
    assert generates, "no generate stages observed"
    last = generates[-1]
    assert last["tokens"] >= 1
    assert last["tok_s"] >= 0.0
    assert last["elapsed_s"] > 0.0


def test_worker_emits_prefill_immediately_even_before_first_token(tmp_path) -> None:
    """prefill 0.0s should fire on submit() entry, NOT only after a token."""
    _app()

    class _BlockedBackend(InferenceBackend):
        def chat(self, *args, **kwargs):  # noqa: ANN001, ANN201
            raise RuntimeError("not used")

        def stream(self, *args, **kwargs):  # noqa: ANN001, ANN201
            # Empty generator — no tokens at all. The worker must still
            # emit prefill so the GUI knows we got the request.
            raise RuntimeError("simulated empty stream")
            yield  # pragma: no cover

        def summarize(self, *args, **kwargs) -> str:  # noqa: ANN001
            return ""

        def is_available(self) -> bool:
            return True

        def list_models(self) -> list[str]:
            return []

    config = AgentConfig(
        workspace=tmp_path,
        model="llama3.2:1b",
        backend="ollama",
        max_iterations=1,
    )
    agent = Agent(config, backend=_BlockedBackend())
    worker = AgentWorker(agent)
    seen: list[tuple[str, dict]] = []
    worker.progress_changed.connect(lambda stage, info: seen.append((stage, info)))
    worker.submit("hi")
    _drain_qt_events(0.05)
    assert seen, "no progress events emitted at all"
    # The very first progress event must be 'prefill', not 'idle' or 'done'.
    assert seen[0][0] == "prefill", f"first event was {seen[0][0]!r}, expected prefill"
