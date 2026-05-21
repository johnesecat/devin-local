"""Agent cancellation + Composer Stop button tests.

The cancel path matters because the user complained about minutes of
idle waiting when the model takes forever to prefill on CPU. The Stop
button gives them a way out without killing the process.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from devin_local.agent import Agent, AgentConfig
from devin_local.gui.widgets import Composer
from devin_local.inference.backend import (
    ChatChunk,
    ChatResponse,
    InferenceBackend,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _SlowBackend(InferenceBackend):
    """A backend whose chat()/stream_chat() blocks until cancel arrives."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self.closed = False
        self._started = threading.Event()

    def chat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        self._started.set()
        # Block until the cancel signal arrives or 10s elapses (safety net).
        self._cancel.wait(timeout=10.0)
        # Honest behavior: a torn-down stream raises.
        raise RuntimeError("backend closed by cancel")

    def stream_chat(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        self._started.set()
        self._cancel.wait(timeout=10.0)
        raise RuntimeError("backend closed by cancel")
        yield ChatChunk(delta="", done=False)  # pragma: no cover

    def close(self) -> None:
        self.closed = True
        self._cancel.set()

    def summarize(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return []


def test_agent_request_cancel_sets_flag_and_closes_backend(tmp_path) -> None:
    backend = _SlowBackend()
    config = AgentConfig(
        workspace=tmp_path,
        model="llama3.2:1b",
        backend="ollama",
        max_iterations=5,
    )
    agent = Agent(config, backend=backend)

    assert agent.cancel_requested() is False
    agent.request_cancel()
    assert agent.cancel_requested() is True
    assert backend.closed is True
    agent.reset_cancel()
    assert agent.cancel_requested() is False


def test_agent_handle_user_returns_cancelled_when_cancelled_during_inference(tmp_path) -> None:
    """handle_user() should return a 'cancelled' AgentTurn after a cancel
    rather than raising, so the GUI can render a friendly notice."""
    backend = _SlowBackend()
    config = AgentConfig(
        workspace=tmp_path,
        model="llama3.2:1b",
        backend="ollama",
        max_iterations=3,
    )
    agent = Agent(config, backend=backend)

    result: list[Any] = []

    def _run() -> None:
        try:
            turn = agent.handle_user("hello", stream=False)
            result.append(turn)
        except Exception as exc:  # pragma: no cover - shouldn't happen
            result.append(exc)

    t = threading.Thread(target=_run)
    t.start()
    assert backend._started.wait(timeout=5.0), "backend never started"
    # Give the agent loop a beat to enter the inference call.
    time.sleep(0.05)
    agent.request_cancel()
    t.join(timeout=5.0)
    assert not t.is_alive(), "handle_user did not return after cancel"
    assert result, "no result emitted"
    assert not isinstance(result[0], Exception), result[0]
    turn = result[0]
    assert "cancelled" in turn.assistant_text.lower()


def test_composer_shows_stop_button_when_busy() -> None:
    _app()
    composer = Composer()
    # Initial state: Send visible, Stop hidden.
    assert composer._send.isVisible() or not composer.isVisible()
    # Becoming busy hides Send and shows Stop.
    composer.set_busy(True)
    composer.show()
    QApplication.processEvents()
    assert composer._stop.isVisibleTo(composer) is True
    assert composer._send.isVisibleTo(composer) is False
    # Going idle reverses it.
    composer.set_busy(False)
    QApplication.processEvents()
    assert composer._send.isVisibleTo(composer) is True
    assert composer._stop.isVisibleTo(composer) is False


def test_composer_stop_button_emits_cancel_signal() -> None:
    _app()
    composer = Composer()
    triggered: list[bool] = []
    composer.cancel_requested.connect(lambda: triggered.append(True))
    composer.set_busy(True)
    composer._stop.click()
    assert triggered == [True]
