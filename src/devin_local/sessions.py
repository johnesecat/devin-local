"""Per-session storage layer.

Each devin-local session has THREE files on disk:

- ``<sessions_dir>/<id>.json``   — :class:`SessionInfo` (config + overrides)
- ``<sessions_dir>/<id>.jsonl``  — full transcript (one chat message per line)
- ``<sessions_dir>/<id>.title``  — short, human-friendly title (optional cache)

The GUI's per-session settings panel writes to the JSON file. The agent's
:class:`SessionStore` writes to the JSONL file. Both share the same ``id``.

The sessions root defaults to ``<workspace>/.devin-local/sessions/`` so
sessions move with the workspace. Use :func:`global_sessions_dir` for the
cross-workspace fallback.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "session"


def sessions_dir(workspace: Path) -> Path:
    """Workspace-scoped sessions directory; created if needed."""
    path = workspace / ".devin-local" / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def global_sessions_dir() -> Path:
    """Cross-workspace fallback (used when no workspace is set)."""
    from devin_local.settings import settings_dir

    path = settings_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class SessionInfo:
    """Per-session settings + identity.

    ``None`` valued override fields mean "use the global default at runtime".
    The GUI's per-session settings dialog edits this dataclass directly.
    """

    id: str
    name: str = "New session"
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = field(default_factory=time.time)

    # Per-session agent overrides — None means "inherit from global Settings".
    system_prompt_override: str = ""
    model: str | None = None
    backend: str | None = None
    knowledge_dir: str | None = None  # absolute path
    enable_obliteratus: bool | None = None
    enable_planning: bool | None = None
    parallel_tool_calls: bool | None = None
    max_iterations: int | None = None
    temperature: float | None = None

    # Free-form notes the operator can jot down in the dialog.
    notes: str = ""

    @classmethod
    def new(cls, name: str = "New session", **overrides: object) -> SessionInfo:
        sid = f"{int(time.time())}-{_slugify(name)}-{uuid.uuid4().hex[:6]}"
        info = cls(id=sid, name=name)
        for key, value in overrides.items():
            if hasattr(info, key):
                setattr(info, key, value)
        return info

    def transcript_path(self, sdir: Path) -> Path:
        return sdir / f"{self.id}.jsonl"

    def config_path(self, sdir: Path) -> Path:
        return sdir / f"{self.id}.json"


def _atomic_write_json(path: Path, payload: dict, *, mode: int = 0o600) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(tmp, mode)
    tmp.replace(path)


@dataclass
class SessionManager:
    """List / create / load / save / delete sessions on disk."""

    root: Path

    @classmethod
    def for_workspace(cls, workspace: Path) -> SessionManager:
        return cls(root=sessions_dir(workspace))

    def list(self) -> list[SessionInfo]:
        """Return all known sessions, newest first by ``last_used_ts``."""
        items: list[SessionInfo] = []
        if not self.root.exists():
            return items
        for path in self.root.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                items.append(SessionInfo(**{**asdict(SessionInfo(id="")), **raw}))
            except TypeError:
                continue
        items.sort(key=lambda s: s.last_used_ts, reverse=True)
        return items

    def create(self, name: str = "New session", **overrides: object) -> SessionInfo:
        info = SessionInfo.new(name=name, **overrides)
        self.save(info)
        # Ensure the transcript file exists so the agent can append to it.
        info.transcript_path(self.root).touch(exist_ok=True)
        return info

    def save(self, info: SessionInfo) -> Path:
        info.last_used_ts = time.time()
        path = info.config_path(self.root)
        _atomic_write_json(path, asdict(info))
        return path

    def load(self, session_id: str) -> SessionInfo | None:
        path = self.root / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return SessionInfo(**{**asdict(SessionInfo(id="")), **raw})
        except TypeError:
            return None

    def delete(self, session_id: str) -> bool:
        cfg = self.root / f"{session_id}.json"
        txn = self.root / f"{session_id}.jsonl"
        gone = False
        for path in (cfg, txn):
            if path.exists():
                with contextlib.suppress(OSError):
                    path.unlink()
                gone = True
        return gone

    def touch(self, session_id: str) -> None:
        info = self.load(session_id)
        if info is None:
            return
        info.last_used_ts = time.time()
        self.save(info)

    def transcript_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"
