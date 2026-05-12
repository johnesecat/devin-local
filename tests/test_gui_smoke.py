"""Offscreen smoke tests for the PySide6 GUI.

These tests skip if PySide6 isn't installed (the [gui] extra is optional).
They run with the offscreen Qt platform plugin so they work in CI without
a real display.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="GUI tests require PySide6 ([gui] extra)")

# Force offscreen *before* importing any Qt module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication  # noqa: E402
except ImportError as exc:
    # PySide6 wheels need libEGL/libxcb at runtime on Linux. CI environments
    # without those system libs should skip these tests rather than error.
    pytest.skip(f"PySide6 native libs not available: {exc}", allow_module_level=True)

from devin_local.gui.app import MainWindow  # noqa: E402
from devin_local.gui.widgets import ChatPane, Composer  # noqa: E402
from devin_local.tools.base import ToolResult  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_chat_pane_renders_user_and_assistant(qapp, tmp_path: Path) -> None:
    pane = ChatPane()
    user = pane.add_user("hi")
    assistant = pane.start_assistant()
    pane.append_assistant_delta("Hel")
    pane.append_assistant_delta("lo")
    pane.finish_assistant()
    assert user.text() == "hi"
    assert assistant.text() == "Hello"


def test_chat_pane_renders_tool_card(qapp, tmp_path: Path) -> None:
    pane = ChatPane()
    card = pane.add_tool("write_file", {"path": "x"})
    card.finish(ToolResult(ok=True, output='{"ok": true}'))


def test_composer_emits_submitted_on_send(qapp) -> None:
    composer = Composer()
    received: list[str] = []
    composer.submitted.connect(received.append)
    composer._input.setPlainText("hello there")
    composer._emit()
    assert received == ["hello there"]
    # After submit, the box should be cleared.
    assert composer._input.toPlainText() == ""


def test_main_window_builds_and_lists_all_backends(qapp, tmp_path: Path) -> None:
    win = MainWindow(workspace=tmp_path, backend="ollama", model="llama3.1:8b")
    try:
        backends_in_combo = [
            win._backend_combo.itemData(i) for i in range(win._backend_combo.count())
        ]
        assert set(backends_in_combo) == {"ollama", "layered", "hf"}
        assert "devin-local" in win.windowTitle()
    finally:
        win.close()


def test_shutdown_agent_disconnects_request_submit_from_old_worker(qapp, tmp_path: Path) -> None:
    """_shutdown_agent must disconnect request_submit from the old worker's
    submit slot, otherwise each backend/workspace switch leaks the previous
    worker via a dangling signal connection (Devin Review BUG_0001)."""
    win = MainWindow(workspace=tmp_path, backend="ollama", model="llama3.1:8b")
    try:
        worker_a = win._worker
        assert worker_a is not None
        win._shutdown_agent()
        # After the fix, the connection was removed. Trying to disconnect
        # again must emit a "Failed to disconnect" RuntimeWarning. Before
        # the fix, this disconnect would succeed silently because the
        # leftover connection was never cleaned up.
        with pytest.warns(RuntimeWarning, match="Failed to disconnect"):
            win.request_submit.disconnect(worker_a.submit)
    finally:
        win.close()
