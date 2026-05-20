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


def test_message_bubble_renders_code_blocks(qapp, tmp_path: Path) -> None:
    """A bubble with a ```lang fenced block should split it into a dedicated
    code-block child widget (so syntax highlighting + copy button can apply).
    """
    from devin_local.gui.syntax import CodeBlockWidget
    from devin_local.gui.widgets import MessageBubble

    text = "Here:\n```python\nprint('hi')\n```\nDone."
    bubble = MessageBubble("assistant", text)
    children = bubble.findChildren(CodeBlockWidget)
    assert len(children) == 1
    assert children[0].lang == "python"
    assert "print" in children[0].code


def test_tool_card_for_write_file_renders_file_preview(qapp, tmp_path: Path) -> None:
    """write_file cards should resolve the path against workspace and embed
    a CodeBlockWidget when the file exists on disk."""
    from devin_local.gui.syntax import CodeBlockWidget
    from devin_local.gui.widgets import ToolCard

    target = tmp_path / "hello.py"
    target.write_text("print('hi')\n", encoding="utf-8")
    card = ToolCard("write_file", {"path": "hello.py"}, workspace=tmp_path)
    card.finish(ToolResult(ok=True, output='{"ok": true, "path": "hello.py"}'))
    code_blocks = card.findChildren(CodeBlockWidget)
    assert len(code_blocks) >= 1
    assert "print" in code_blocks[0].code


def test_plan_pane_renders_steps(qapp) -> None:
    from devin_local.agent_planning import PlanStep
    from devin_local.gui.sandbox_panel import PlanPane

    pane = PlanPane()
    pane.set_plan(
        [
            PlanStep(text="A", status="completed"),
            PlanStep(text="B", status="in_progress"),
            PlanStep(text="C", status="pending"),
        ]
    )
    # Visible row text should reflect statuses.
    assert pane._list.count() == 3
    assert "A" in pane._list.item(0).text()
    assert "B" in pane._list.item(1).text()


def test_sandbox_panel_renders_workspace_and_flags(qapp, tmp_path: Path) -> None:
    from devin_local.gui.sandbox_panel import SandboxPanel

    panel = SandboxPanel()
    panel.set_workspace(tmp_path)
    panel.set_terminal_state(str(tmp_path), "ls -la")
    panel.set_desktop_state(True, "screenshot")
    panel.set_tools(["read_file", "write_file", "shell_exec"])
    panel.set_flags(network=True, browser=False, desktop=True)
    assert str(tmp_path) in panel._workspace_lbl.text()
    assert "ls -la" in panel._term_lbl.text()
    assert "3" in panel._tools_lbl.text()


def test_model_selector_populates_combo(qapp) -> None:
    from devin_local.gui.model_selector import ModelSelector

    sel = ModelSelector(current="llama3.1:8b")
    sel.set_installed(["llama3.1:8b", "qwen2.5:7b", "mistral:7b"])
    assert sel.current() == "llama3.1:8b"
    items = [sel._combo.itemData(i) for i in range(sel._combo.count())]
    assert "llama3.1:8b" in items
    assert "qwen2.5:7b" in items


def test_model_selector_keeps_uninstalled_current_with_marker(qapp) -> None:
    from devin_local.gui.model_selector import ModelSelector

    sel = ModelSelector(current="ghost-model:8b")
    sel.set_installed(["llama3.1:8b"])
    items = [sel._combo.itemText(i) for i in range(sel._combo.count())]
    # The previous-but-uninstalled model should still appear so the user
    # knows what's missing.
    assert any("ghost-model:8b" in t for t in items)


def test_main_window_exposes_model_selector_and_plan_pane(qapp, tmp_path: Path) -> None:
    win = MainWindow(workspace=tmp_path, backend="ollama", model="llama3.1:8b")
    try:
        assert win._model_selector.current() == "llama3.1:8b"
        # Plan pane starts empty.
        assert win._plan_pane._list.count() == 0
    finally:
        win.close()


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


def test_message_bubble_has_copy_button_timestamp_and_role(qapp) -> None:
    """The polished assistant bubble must surface a role label, timestamp,
    and a copy-on-hover button so the user can grab the response cleanly."""
    from devin_local.gui.widgets import MessageBubble

    bubble = MessageBubble("assistant", "hello")
    assert bubble._avatar.text() == "devin-local"
    # HH:MM is 5 chars; allow either to permit single-digit hours.
    assert len(bubble._timestamp.text()) in (4, 5)
    assert bubble._copy_btn.text() == "Copy"
    bubble._copy_to_clipboard()
    assert bubble._copy_btn.text() == "Copied"


def test_message_bubble_user_role_uses_user_avatar(qapp) -> None:
    from devin_local.gui.widgets import MessageBubble

    bubble = MessageBubble("user", "do the thing")
    assert bubble._avatar.text() == "You"
    assert bubble._avatar.objectName() == "BubbleAvatarUser"


def test_parse_list_dir_output_handles_files_and_dirs() -> None:
    """The chat tree-card depends on parsing list_dir output reliably."""
    from devin_local.gui.widgets import parse_list_dir_output

    raw = "dir         0  src\nfile      1234  README.md\nfile        12  .gitignore"
    entries = parse_list_dir_output(raw)
    assert entries == [
        ("dir", "src", 0),
        ("file", "README.md", 1234),
        ("file", ".gitignore", 12),
    ]


def test_parse_list_dir_output_handles_empty_marker() -> None:
    from devin_local.gui.widgets import parse_list_dir_output

    assert parse_list_dir_output("(empty)") == []
    assert parse_list_dir_output("") == []


def test_parse_find_files_output_returns_file_entries() -> None:
    from devin_local.gui.widgets import parse_find_files_output

    raw = "src/devin_local/cli.py\nsrc/devin_local/gui/app.py\nREADME.md"
    entries = parse_find_files_output(raw)
    assert entries == [
        ("file", "src/devin_local/cli.py", 0),
        ("file", "src/devin_local/gui/app.py", 0),
        ("file", "README.md", 0),
    ]


def test_parse_find_files_output_handles_no_matches() -> None:
    from devin_local.gui.widgets import parse_find_files_output

    assert parse_find_files_output("(no matches)") == []


def test_tool_card_for_list_dir_renders_folder_tree(qapp, tmp_path: Path) -> None:
    """list_dir cards must render a real QTreeWidget, not just text."""
    from PySide6.QtWidgets import QTreeWidget

    from devin_local.gui.widgets import ToolCard

    raw = "dir         0  src\nfile      1234  README.md"
    card = ToolCard("list_dir", {"path": "."}, workspace=tmp_path)
    card.finish(ToolResult(ok=True, output=raw))
    trees = card.findChildren(QTreeWidget)
    assert len(trees) == 1
    tree = trees[0]
    assert tree.objectName() == "FolderTree"
    # Top-level row count should match the number of parsed entries.
    assert tree.topLevelItemCount() == 2


def test_tool_card_for_find_files_groups_entries_by_directory(qapp, tmp_path: Path) -> None:
    """find_files results that contain paths should group by parent dir so
    the chat shows a real visual tree (not a flat list)."""
    from PySide6.QtWidgets import QTreeWidget

    from devin_local.gui.widgets import ToolCard

    raw = "src/devin_local/cli.py\nsrc/devin_local/gui/app.py\nREADME.md"
    card = ToolCard("find_files", {"pattern": "**/*.py", "root": "."}, workspace=tmp_path)
    card.finish(ToolResult(ok=True, output=raw))
    tree = card.findChildren(QTreeWidget)[0]
    # 2 distinct parent directories ("src/devin_local", "src/devin_local/gui")
    # plus 1 root-level entry (README.md) = at least 3 top-level rows.
    top_texts = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert any("src/devin_local" in t for t in top_texts)
    assert any("README.md" in t for t in top_texts)


def test_tool_card_for_read_file_includes_path_meta(qapp, tmp_path: Path) -> None:
    """read_file cards must show the path + a syntax-highlighted preview of
    the returned contents (so users can see what the model actually read)."""
    from devin_local.gui.syntax import CodeBlockWidget
    from devin_local.gui.widgets import ToolCard

    target = tmp_path / "hello.py"
    target.write_text("print('hi')\n", encoding="utf-8")
    card = ToolCard("read_file", {"path": "hello.py"}, workspace=tmp_path)
    card.finish(ToolResult(ok=True, output="print('hi')\n"))
    blocks = card.findChildren(CodeBlockWidget)
    assert any("print" in b.code for b in blocks)


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
