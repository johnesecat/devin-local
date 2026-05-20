"""Offscreen smoke tests for the settings dialog."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Skip the whole module if Qt isn't importable for some reason.
pytest.importorskip("PySide6")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "devin-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(home))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return home


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_settings_dialog_has_all_tabs() -> None:
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    expected = [
        "General",
        "MCP",
        "GitHub",
        "Backends",
        "Knowledge",
        "Figma",
        "Appearance",
    ]
    actual = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
    assert actual == expected


def test_general_tab_round_trips_through_settings_load() -> None:
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog
    from devin_local.settings import Settings

    dlg = SettingsDialog()
    dlg.general_tab.default_model.setText("custom-model:42")
    dlg.general_tab.default_backend.setCurrentText("layered")
    dlg.general_tab.parallel.setChecked(False)
    dlg.general_tab.obliteratus.setChecked(False)
    dlg._on_save()

    loaded = Settings.load()
    assert loaded.general.default_model == "custom-model:42"
    assert loaded.general.default_backend == "layered"
    assert loaded.general.parallel_tool_calls is False
    assert loaded.general.enable_obliteratus is False


def test_mcp_tab_add_and_remove_persists() -> None:
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog
    from devin_local.settings import MCPServer, load_mcp_servers

    dlg = SettingsDialog()
    dlg.mcp_tab._servers.append(MCPServer(name="test-server", command="/usr/bin/echo", args=["hi"]))
    dlg.mcp_tab._refresh()
    dlg._on_save()

    saved = load_mcp_servers()
    assert any(s.name == "test-server" for s in saved)


def test_github_tab_status_label_shows_not_verified_initially() -> None:
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    assert "Not verified" in dlg.github_tab.status.text()


def test_appearance_tab_persists_font_scale() -> None:
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog
    from devin_local.settings import Settings

    dlg = SettingsDialog()
    dlg.appearance_tab.font_scale.setValue(1.25)
    dlg._on_save()

    loaded = Settings.load()
    assert loaded.appearance.font_scale == pytest.approx(1.25)


def test_backends_tab_lists_three_backends() -> None:
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    assert set(dlg.backends_tab._rows.keys()) == {"ollama", "layered", "hf"}


def test_knowledge_tab_upload_file_persists_to_user_store(tmp_path: Path) -> None:
    """Programmatic add via the Knowledge tab lands in the user knowledge store."""
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog
    from devin_local.knowledge.store import KnowledgeStore
    from devin_local.settings import user_knowledge_path

    dlg = SettingsDialog()
    # Drive the tab's store directly (the QFileDialog is offscreen-unfriendly).
    tab = dlg.knowledge_tab
    tab._store.add(title="GUI upload", body="contents from uploaded file")
    tab._refresh()
    tab.knowledge_changed.emit()

    # Item shows up in the list with the right title.
    items = [tab.list.item(i).text() for i in range(tab.list.count())]
    assert any("GUI upload" in t for t in items)

    # And the underlying file persists for the next session.
    persisted = KnowledgeStore.open(user_knowledge_path())
    assert any(n.title == "GUI upload" for n in persisted.all())


def test_knowledge_changed_signal_fires_on_add() -> None:
    """Adding a note triggers ``SettingsDialog.knowledge_changed`` for the main window."""
    _app()
    from devin_local.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    received: list[bool] = []
    dlg.knowledge_changed.connect(lambda: received.append(True))
    dlg.knowledge_tab._store.add(title="Note", body="body")
    dlg.knowledge_tab._refresh()
    dlg.knowledge_tab.knowledge_changed.emit()
    assert received == [True]
