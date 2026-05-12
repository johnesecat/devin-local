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

from PySide6.QtWidgets import QApplication  # noqa: E402

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
