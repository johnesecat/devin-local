"""Persistent user settings for devin-local.

Qt-free so the CLI and tests can read/write the same store without dragging
in PySide6. The Settings dialog (``gui/settings_dialog.py``) is a thin Qt
wrapper around this module.

Storage layout::

    ~/.devin-local/settings.json     # general + appearance + backend prefs
    ~/.devin-local/mcp_servers.json  # MCP server list (CRUD'd via the GUI)
    ~/.devin-local/github.json       # GitHub PAT; 0600 perms

The directory is created with ``0700`` perms and individual secret-bearing
files with ``0600`` perms. On Windows ``chmod`` is a no-op; access control
is left to NTFS defaults (the dir is under the user's profile).
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_THEME = "tokyo-night"
DEFAULT_FONT_SCALE = 1.0
DEFAULT_BACKEND = "ollama"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def _hw_recommended_default_model() -> str:
    """Probe hardware and recommend a default model. Empty string on failure.

    Imported lazily so the ``settings`` module stays Qt/torch/etc. free at
    import time — the agent CLI imports settings before anything else.
    """
    try:
        from devin_local.hardware import recommend_default_model

        return recommend_default_model()
    except Exception:  # noqa: BLE001
        return ""


def settings_dir() -> Path:
    """Return the on-disk settings directory, creating it if needed.

    Respects ``DEVIN_LOCAL_HOME`` for tests. Otherwise uses
    ``~/.devin-local`` on every platform (keeps it simple; XDG variants on
    Linux can be a follow-up).
    """
    override = os.environ.get("DEVIN_LOCAL_HOME")
    base = Path(override) if override else Path.home() / ".devin-local"
    base.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(base, 0o700)
    return base


def _atomic_write(path: Path, content: str, *, secret: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    if secret:
        with contextlib.suppress(OSError, NotImplementedError):
            os.chmod(tmp, 0o600)
    tmp.replace(path)


@dataclass
class GeneralSettings:
    workspace: str = ""  # empty => "use cwd"
    default_model: str = DEFAULT_MODEL
    default_backend: str = DEFAULT_BACKEND
    ollama_host: str = DEFAULT_OLLAMA_HOST
    parallel_tool_calls: bool = True
    enable_obliteratus: bool = True
    enable_planning: bool = True
    # When True, render the full upstream-style verbose system prompt
    # (~13 KB). When False (default), use the slim ~3 KB prompt — faster
    # local inference, same identity / honesty / planning rules.
    verbose_prompt: bool = False
    # When True, ship every registered tool to the model on every turn
    # (the "full toolbelt"). When False (default), the agent picks a
    # relevant subset based on keyword overlap with the user's message,
    # which can shave kilobytes off prefill cost on CPU. The "always-
    # include" set (read/write/edit/list/shell/knowledge) stays resident
    # in both modes so foundational tools never round-trip.
    full_toolbelt: bool = False


@dataclass
class AppearanceSettings:
    theme: str = DEFAULT_THEME
    font_scale: float = DEFAULT_FONT_SCALE
    accent: str = "#7aa2f7"
    show_sandbox_panel: bool = True
    show_plan_pane: bool = True


@dataclass
class Settings:
    general: GeneralSettings = field(default_factory=GeneralSettings)
    appearance: AppearanceSettings = field(default_factory=AppearanceSettings)

    @classmethod
    def load(cls) -> Settings:
        path = settings_dir() / "settings.json"
        if not path.exists():
            # First launch: pick a model that's actually realistic for this
            # box. Saves the user from getting an 8 B model on a 4-core
            # 8 GB VM and waiting ten minutes for the first token.
            inst = cls()
            recommended = _hw_recommended_default_model()
            if recommended:
                inst.general.default_model = recommended
            return inst
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls(
            general=GeneralSettings(**{**asdict(GeneralSettings()), **raw.get("general", {})}),
            appearance=AppearanceSettings(
                **{**asdict(AppearanceSettings()), **raw.get("appearance", {})}
            ),
        )

    def save(self) -> Path:
        path = settings_dir() / "settings.json"
        _atomic_write(
            path,
            json.dumps(
                {"general": asdict(self.general), "appearance": asdict(self.appearance)},
                indent=2,
            ),
        )
        return path


@dataclass
class MCPServer:
    name: str
    command: str = ""  # for stdio servers
    args: list[str] = field(default_factory=list)
    url: str = ""  # for HTTP/SSE servers
    transport: str = "stdio"  # "stdio" | "http" | "sse"
    enabled: bool = True
    env: dict[str, str] = field(default_factory=dict)


def _mcp_path() -> Path:
    return settings_dir() / "mcp_servers.json"


def load_mcp_servers() -> list[MCPServer]:
    path = _mcp_path()
    if not path.exists():
        # First run: seed from the repo's mcp_servers.json if present, else empty.
        repo_seed = Path.cwd() / "mcp_servers.json"
        if repo_seed.exists() and repo_seed != path:
            try:
                raw = json.loads(repo_seed.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {"servers": []}
        else:
            raw = {"servers": []}
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {"servers": []}
    servers: list[MCPServer] = []
    for entry in raw.get("servers", []):
        if not isinstance(entry, dict):
            continue
        try:
            servers.append(MCPServer(**{**asdict(MCPServer(name="")), **entry}))
        except TypeError:
            continue
    return servers


def save_mcp_servers(servers: list[MCPServer]) -> Path:
    path = _mcp_path()
    _atomic_write(
        path,
        json.dumps({"servers": [asdict(s) for s in servers]}, indent=2),
    )
    return path


@dataclass
class GitHubSettings:
    pat: str = ""
    username: str = ""
    last_verified_login: str = ""  # cached login from /user, for display only


def _github_path() -> Path:
    return settings_dir() / "github.json"


def load_github() -> GitHubSettings:
    path = _github_path()
    if not path.exists():
        return GitHubSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return GitHubSettings()
    return GitHubSettings(**{**asdict(GitHubSettings()), **raw})


def save_github(settings: GitHubSettings) -> Path:
    path = _github_path()
    _atomic_write(path, json.dumps(asdict(settings), indent=2), secret=True)
    return path


def user_tools_dir() -> Path:
    """Return the user-level Python-plugin tool directory.

    Drop a ``.py`` file here that calls ``@tool`` and the Agent loads it
    on the next ``ensure_initialized()``. The GUI's Settings → Tools tab
    creates / edits / deletes files inside this directory.
    """
    tdir = settings_dir() / "tools"
    tdir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(tdir, 0o700)
    return tdir


def user_knowledge_path() -> Path:
    """Return the user-level (not workspace-scoped) knowledge store path.

    Uploads made via the GUI's Settings → Knowledge tab land here so they
    persist across workspaces. The Agent merges this store with the
    workspace-local ``<workspace>/knowledge/store.jsonl`` at session start
    and embeds both into the system prompt — they are NOT searched per
    turn.
    """
    kdir = settings_dir() / "knowledge"
    kdir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(kdir, 0o700)
    return kdir / "store.jsonl"


def verify_github_pat(pat: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """Hit ``GET /user`` with the PAT. Returns the decoded JSON on success.

    Raises ``RuntimeError`` on any non-200. Never logs the PAT.
    """
    import httpx

    if not pat:
        raise RuntimeError("no PAT provided")
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
    if response.status_code != 200:
        raise RuntimeError(f"GitHub returned {response.status_code}; check the PAT and its scopes")
    return response.json()
