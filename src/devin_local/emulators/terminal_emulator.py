"""Persistent shell session emulator.

We can't reliably use a PTY on Windows without `pywinpty`/`ConPTY`, so this
emulator runs the shell as a long-lived subprocess with `stdin`/`stdout`
pipes. A background thread drains stdout into a queue so the agent can read
incremental output via `read()`. This is "good enough" for command-line
workflows; for full terminal control (cursor addressing, color rendering)
users should install `pywinpty` and pass `use_pty=True`.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _shell_command() -> list[str]:
    """Pick the persistent-shell command for the current OS."""
    if sys.platform.startswith("win"):
        # Use PowerShell with no profile + an interactive command loop.
        # `-NoExit -Command -` keeps reading PowerShell commands from stdin.
        for candidate in ("pwsh.exe", "powershell.exe"):
            from shutil import which

            if which(candidate):
                return [candidate, "-NoLogo", "-NoProfile", "-NoExit", "-Command", "-"]
        return ["cmd.exe", "/Q", "/K"]
    return [os.environ.get("SHELL", "/bin/bash"), "-i"]


@dataclass
class _Session:
    """Internal state for one persistent shell."""

    sid: str
    proc: subprocess.Popen[bytes]
    workspace: Path
    out_queue: queue.Queue[bytes] = field(default_factory=queue.Queue)
    reader_thread: threading.Thread | None = None
    closed: bool = False


class TerminalEmulator:
    """Manages a pool of persistent shell sessions identified by string ids."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def open_session(self, workspace: Path) -> str:
        """Spawn a new shell and return its session id."""
        sid = uuid.uuid4().hex[:8]
        cmd = _shell_command()
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd,
            cwd=str(workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            creationflags=creationflags,
        )
        session = _Session(sid=sid, proc=proc, workspace=workspace)
        thread = threading.Thread(
            target=self._drain, args=(session,), name=f"shell-reader-{sid}", daemon=True
        )
        session.reader_thread = thread
        thread.start()
        with self._lock:
            self._sessions[sid] = session
        return sid

    def list_sessions(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions.keys())

    def send(self, sid: str, command: str) -> None:
        """Send `command` (followed by newline) to the session's stdin."""
        session = self._get(sid)
        if session.proc.stdin is None:
            raise RuntimeError(f"Session {sid} has no stdin pipe.")
        line = command.rstrip("\n") + os.linesep
        session.proc.stdin.write(line.encode("utf-8", errors="replace"))
        session.proc.stdin.flush()

    def read(self, sid: str, timeout: float = 5.0) -> str:
        """Drain pending output for up to `timeout` seconds."""
        session = self._get(sid)
        chunks: list[bytes] = []
        deadline = _monotonic() + timeout
        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            try:
                chunk = session.out_queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if not chunks:
                    continue
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def close(self, sid: str) -> None:
        """Terminate a session and free its resources."""
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is None:
            return
        session.closed = True
        try:
            if session.proc.stdin and not session.proc.stdin.closed:
                session.proc.stdin.close()
        except OSError:
            pass
        try:
            session.proc.terminate()
            session.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            session.proc.kill()
        except OSError:
            pass

    def close_all(self) -> None:
        for sid in self.list_sessions():
            self.close(sid)

    # ---------- internals ----------

    def _get(self, sid: str) -> _Session:
        with self._lock:
            session = self._sessions.get(sid)
        if session is None or session.closed:
            raise RuntimeError(f"Unknown shell session: {sid}")
        return session

    @staticmethod
    def _drain(session: _Session) -> None:
        assert session.proc.stdout is not None
        try:
            while not session.closed:
                chunk = session.proc.stdout.read(4096)
                if not chunk:
                    break
                session.out_queue.put(chunk)
        except (OSError, ValueError):
            pass


def _monotonic() -> float:
    import time

    return time.monotonic()


# Re-export for tests / external callers without leaking the dataclass.
def session_count(emulator: TerminalEmulator) -> int:
    return len(emulator.list_sessions())


__all__ = ["TerminalEmulator", "session_count"]


# Help mypy not complain about the unused Any import (kept for forward use).
_ = Any
