"""Background worker that runs the agent off the Qt event loop.

PySide6's event loop must not block, so model inference (which can take
minutes on layered/HF backends) runs on a worker thread. The worker emits
Qt signals for token deltas, tool events, and turn completion; the UI just
connects slots to those signals.
"""

from __future__ import annotations

import contextlib
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

from devin_local.agent import Agent
from devin_local.agent_planning import Plan
from devin_local.inference.backend import ChatChunk
from devin_local.tools.base import ToolResult


class AgentWorker(QObject):
    """Owns an :class:`Agent` and runs one user turn per request.

    Signals:
      - ``token``: incremental token text (str)
      - ``tool_started``: tool name + arguments (str, dict)
      - ``tool_finished``: tool name, arguments, result (str, dict, ToolResult)
      - ``plan_updated``: snapshot of the current agent plan (object)
      - ``turn_finished``: final assistant text + tool count + elapsed seconds
      - ``error``: backend / agent exception (str)
    """

    token = Signal(str)
    tool_started = Signal(str, dict)
    tool_finished = Signal(str, dict, object)  # ToolResult
    plan_updated = Signal(object)  # Plan
    turn_finished = Signal(str, int, float)
    error = Signal(str)
    state_changed = Signal(str)  # "idle" | "thinking" | "tool" | "error" | "cancelled"
    cancelled = Signal()

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent
        self._stream_enabled = True

    @Slot(bool)
    def set_streaming(self, enabled: bool) -> None:
        self._stream_enabled = enabled

    @Slot(str)
    def submit(self, user_text: str) -> None:
        """Run one user turn. Must be invoked via QMetaObject.invokeMethod or
        a queued Qt connection because we'll touch the agent's mutable state.
        """
        import time

        if not user_text.strip():
            return
        self.state_changed.emit("thinking")

        def _on_chunk(chunk: ChatChunk) -> None:
            if not chunk.done and chunk.delta:
                self.token.emit(chunk.delta)

        def _on_tool(name: str, args: dict[str, Any], result: ToolResult) -> None:
            self.tool_finished.emit(name, args, result)

        def _on_tool_start(name: str, args: dict[str, Any]) -> None:
            self.tool_started.emit(name, args)
            self.state_changed.emit("tool")

        def _on_plan(plan: Plan) -> None:
            self.plan_updated.emit(plan)

        self.agent.add_stream_observer(_on_chunk)
        self.agent.add_tool_observer(_on_tool)
        self.agent.add_tool_start_observer(_on_tool_start)
        self.agent.add_plan_observer(_on_plan)
        try:
            start = time.monotonic()
            turn = self.agent.handle_user(user_text, stream=self._stream_enabled)
            elapsed = time.monotonic() - start
            self.turn_finished.emit(turn.assistant_text, len(turn.tool_results), elapsed)
            self.state_changed.emit("idle")
        except Exception as exc:  # noqa: BLE001 - surface to UI, don't crash thread
            self.error.emit(f"{type(exc).__name__}: {exc}")
            self.state_changed.emit("error")
        finally:
            with contextlib.suppress(ValueError):
                self.agent._stream_observers.remove(_on_chunk)
            with contextlib.suppress(ValueError):
                self.agent._observers.remove(_on_tool)
            with contextlib.suppress(ValueError):
                self.agent._tool_start_observers.remove(_on_tool_start)
            with contextlib.suppress(ValueError):
                self.agent._plan_observers.remove(_on_plan)

    @Slot()
    def request_cancel(self) -> None:
        """Cancel the in-flight turn.

        Safe to call from any thread — ``Agent.request_cancel`` sets a
        ``threading.Event`` and closes the inference HTTP connection, which
        unblocks a slow CPU prefill immediately. ``submit`` then returns
        through its normal "cancelled" path.
        """
        with contextlib.suppress(Exception):
            self.agent.request_cancel()
        self.state_changed.emit("cancelled")
        self.cancelled.emit()

    @Slot()
    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self.agent.shutdown()


def run_in_worker_thread(parent: QObject, worker: AgentWorker) -> QThread:
    """Move `worker` to a fresh QThread, start it, and return the thread.

    The caller is responsible for keeping a reference to the returned thread
    so it isn't garbage-collected while it's still running.
    """
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.start()
    return thread
