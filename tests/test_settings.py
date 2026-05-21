"""Tests for the Qt-free settings module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from devin_local.settings import (
    GeneralSettings,
    GitHubSettings,
    MCPServer,
    Settings,
    load_github,
    load_mcp_servers,
    save_github,
    save_mcp_servers,
    settings_dir,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "devin-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(home))
    return home


def test_settings_dir_is_created_and_overridable() -> None:
    path = settings_dir()
    assert path.exists()
    assert path.is_dir()
    assert str(path).endswith("devin-home")


def test_settings_round_trip() -> None:
    s = Settings()
    s.general = GeneralSettings(
        workspace="/tmp/ws",
        default_model="llama3.2:3b",
        default_backend="layered",
        ollama_host="http://127.0.0.1:11434",
        parallel_tool_calls=False,
        enable_obliteratus=False,
        enable_planning=True,
    )
    s.appearance.theme = "tokyo-night"
    s.appearance.font_scale = 1.2
    s.save()

    loaded = Settings.load()
    assert loaded.general.workspace == "/tmp/ws"
    assert loaded.general.default_model == "llama3.2:3b"
    assert loaded.general.default_backend == "layered"
    assert loaded.general.parallel_tool_calls is False
    assert loaded.general.enable_obliteratus is False
    assert loaded.general.enable_planning is True
    assert loaded.appearance.font_scale == pytest.approx(1.2)


def test_settings_load_returns_defaults_when_file_missing() -> None:
    loaded = Settings.load()
    assert isinstance(loaded.general, GeneralSettings)
    assert loaded.general.default_backend == "ollama"
    assert loaded.appearance.theme == "tokyo-night"


def test_settings_load_recovers_from_corrupt_file() -> None:
    (settings_dir() / "settings.json").write_text("{not valid json", encoding="utf-8")
    loaded = Settings.load()
    assert loaded.general.default_backend == "ollama"


def test_mcp_server_crud_round_trip() -> None:
    servers = [
        MCPServer(name="filesystem", command="mcp-fs", args=["--root", "/tmp"]),
        MCPServer(name="web", url="http://127.0.0.1:8080/mcp", transport="http"),
    ]
    save_mcp_servers(servers)
    loaded = load_mcp_servers()
    assert len(loaded) == 2
    assert loaded[0].name == "filesystem"
    assert loaded[0].command == "mcp-fs"
    assert loaded[0].args == ["--root", "/tmp"]
    assert loaded[1].url == "http://127.0.0.1:8080/mcp"
    assert loaded[1].transport == "http"


def test_mcp_server_load_handles_missing_fields_gracefully() -> None:
    (settings_dir() / "mcp_servers.json").write_text(
        json.dumps({"servers": [{"name": "broken"}, {"not": "a-server"}, "junk"]}),
        encoding="utf-8",
    )
    loaded = load_mcp_servers()
    assert len(loaded) == 1
    assert loaded[0].name == "broken"


def test_github_round_trip_and_perms_when_supported() -> None:
    gh = GitHubSettings(pat="ghp_dummy_value", username="jacob")
    path = save_github(gh)
    loaded = load_github()
    assert loaded.pat == "ghp_dummy_value"
    assert loaded.username == "jacob"
    if sys.platform != "win32":
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"github.json should be 0600, got {oct(mode)}"


def test_github_pat_is_never_logged(capsys: pytest.CaptureFixture[str]) -> None:
    gh = GitHubSettings(pat="ghp_super_secret_value_do_not_log", username="jacob")
    save_github(gh)
    captured = capsys.readouterr()
    assert "ghp_super_secret_value_do_not_log" not in captured.out
    assert "ghp_super_secret_value_do_not_log" not in captured.err


def test_atomic_write_does_not_leave_temp_file_on_success() -> None:
    save_github(GitHubSettings(pat="x"))
    leftovers = list(settings_dir().glob("*.tmp"))
    assert leftovers == [], f"leftover tmp files: {leftovers}"
