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
    expected = ["General", "MCP", "GitHub", "Backends", "Appearance"]
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
