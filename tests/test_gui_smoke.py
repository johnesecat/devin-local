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


def test_chat_pane_has_dark_theme_object_names(qapp) -> None:
    """Regression for the 'beige chat under dark sidebar' bug.

    The ChatPane viewport must carry the object names referenced by the
    theme stylesheet so the dark ``bg_0`` rule actually matches it.
    Without these the QScrollArea viewport defaults to a near-white system
    color and the chat looks mismatched with the rest of the dark UI.
    """
    pane = ChatPane()
    assert pane.objectName() == "ChatPane"
    container = pane.widget()
    assert container is not None
    assert container.objectName() == "ChatPaneContainer"
    viewport = pane.viewport()
    assert viewport is not None
    assert viewport.objectName() == "ChatPaneViewport"


def test_chat_pane_theme_rule_targets_object_names() -> None:
    """The dark theme stylesheet must contain a rule that matches the
    ChatPane's object names — otherwise setting them on the widget is a
    no-op."""
    from devin_local.gui.theme import stylesheet

    css = stylesheet()
    assert "QScrollArea#ChatPane" in css
    assert "ChatPaneContainer" in css or "ChatPaneViewport" in css


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


def test_ui_creator_dialog_has_figma_import_button(qapp, tmp_path: Path) -> None:
    """The UI Creator's bottom button row must include an 'Import from Figma'
    entry point so a Figma design can populate the canvas directly."""
    from devin_local.ui_creator.dialog import UiCreatorDialog

    dlg = UiCreatorDialog(workspace=tmp_path)
    try:
        assert hasattr(dlg, "_figma_btn")
        assert "Figma" in dlg._figma_btn.text()
    finally:
        dlg.close()


def test_ui_creator_open_from_main_window_uses_correct_module(qapp, tmp_path: Path) -> None:
    """The main window's _open_ui_creator must import from devin_local.ui_creator,
    not the (non-existent) devin_local.gui.ui_creator path."""
    import inspect

    src = inspect.getsource(MainWindow._open_ui_creator)
    assert "from devin_local.ui_creator import UiCreatorDialog" in src
    assert "from devin_local.gui.ui_creator import" not in src


def test_settings_dialog_has_figma_tab(qapp) -> None:
    """Settings dialog must include a Figma tab with PAT input + Import button."""
    from devin_local.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    try:
        tab_names = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
        assert "Figma" in tab_names
        assert hasattr(dlg.figma_tab, "token")
        assert hasattr(dlg.figma_tab, "import_btn")
        assert hasattr(dlg.figma_tab, "test_btn")
    finally:
        dlg.close()


def test_per_session_settings_round_trip(qapp, tmp_path: Path) -> None:
    """PerSessionSettingsDialog must persist system_prompt_override + knowledge_dir
    into the SessionInfo when the user saves, and that data must survive a
    full save/load cycle through disk via SessionManager."""
    from devin_local.gui.session_settings_dialog import PerSessionSettingsDialog
    from devin_local.sessions import SessionManager
    from devin_local.settings import Settings

    mgr = SessionManager.for_workspace(tmp_path)
    info = mgr.create(name="My session")
    settings = Settings.load()
    dlg = PerSessionSettingsDialog(info, settings)
    try:
        # Drive the dialog state as if the operator typed in the fields.
        dlg._agent_tab.editor.setPlainText("YOU ARE A PIRATE.")
        kb_dir = tmp_path / "kb"
        dlg._knowledge_tab.path_edit.setText(str(kb_dir))
        # Trigger the save flow and capture the emitted SessionInfo.
        captured: list = []
        dlg.session_saved.connect(lambda s: captured.append(s))
        dlg._on_save()
        assert captured, "session_saved did not fire"
        updated = captured[0]
        assert "PIRATE" in updated.system_prompt_override
        assert updated.knowledge_dir == str(kb_dir)
        # Roundtrip through disk too.
        mgr.save(updated)
        reloaded = mgr.load(updated.id)
        assert reloaded is not None
        assert "PIRATE" in reloaded.system_prompt_override
        assert reloaded.knowledge_dir == str(kb_dir)
    finally:
        dlg.close()


def test_per_session_verbose_prompt_toggle_round_trips(qapp, tmp_path: Path) -> None:
    """The Behavior tab's 'Verbose system prompt' override must round-trip
    through SessionInfo so the slim/verbose choice actually persists."""
    from devin_local.gui.session_settings_dialog import PerSessionSettingsDialog
    from devin_local.sessions import SessionManager
    from devin_local.settings import Settings

    mgr = SessionManager.for_workspace(tmp_path)
    info = mgr.create(name="Verbose session")
    settings = Settings.load()
    dlg = PerSessionSettingsDialog(info, settings)
    try:
        # Default: not overridden -> None.
        assert dlg._behavior_tab.values()["verbose_prompt"] is None
        # Flip the override on and check the value.
        dlg._behavior_tab._verbose_row._cb.setChecked(True)
        dlg._behavior_tab._verbose_cb.setChecked(True)
        assert dlg._behavior_tab.values()["verbose_prompt"] is True
        captured: list = []
        dlg.session_saved.connect(lambda s: captured.append(s))
        dlg._on_save()
        assert captured
        assert captured[0].verbose_prompt is True
        # Roundtrip through disk.
        mgr.save(captured[0])
        reloaded = mgr.load(captured[0].id)
        assert reloaded is not None
        assert reloaded.verbose_prompt is True
    finally:
        dlg.close()


def test_settings_dialog_tools_tab_lists_user_tool_files(qapp, tmp_path: Path, monkeypatch) -> None:
    """Settings → Tools tab must list user-defined .py tool files and call
    ``register_user_tools`` on the running agent when the user clicks
    'Reload'. This test only covers the listing behavior; the agent-side
    reload is covered by test_user_tools.py."""
    from devin_local.gui.settings_dialog import SettingsDialog

    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(tmp_path / "home"))
    tools_dir = tmp_path / "home" / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "wordcount.py").write_text(
        "from devin_local.tools.user_tools import tool\n\n"
        '@tool(name="wordcount", description="count words")\n'
        "def wordcount(args):\n"
        '    return str(len(args.get("text", "").split()))\n',
        encoding="utf-8",
    )
    dlg = SettingsDialog()
    try:
        items = [dlg.tools_tab._list.item(i).text() for i in range(dlg.tools_tab._list.count())]
        assert any("wordcount.py" in t and "wordcount" in t for t in items)
    finally:
        dlg.close()


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
