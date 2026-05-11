"""Terminal / shell execution tools.

The agent gets two related tools:

- `shell_exec`: run a one-shot command in the workspace's default shell and
  return its output. Handles timeouts and captures stdout+stderr.
- `shell_session_*`: open / send / read / close a persistent shell session
  through the `TerminalEmulator` (see `devin_local.emulators.terminal_emulator`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from devin_local.emulators.terminal_emulator import TerminalEmulator
from devin_local.tools.base import (
    Tool,
    ToolResult,
    error_result,
    optional_str,
    require_str,
    text_result,
    truncate,
)


def _default_shell() -> list[str]:
    """Return the default shell command list for the current OS.

    On Windows we prefer PowerShell 7 (`pwsh`) when on PATH, else
    PowerShell 5.1 (`powershell`), else `cmd`. On Unix we use the
    user's `$SHELL` or `/bin/bash`.
    """
    if sys.platform.startswith("win"):
        for candidate in ("pwsh.exe", "powershell.exe", "cmd.exe"):
            if _which(candidate):
                if candidate == "cmd.exe":
                    return ["cmd.exe", "/d", "/s", "/c"]
                return [candidate, "-NoLogo", "-NoProfile", "-Command"]
        return ["cmd.exe", "/d", "/s", "/c"]
    import os

    shell = os.environ.get("SHELL", "/bin/bash")
    return [shell, "-lc"]


def _which(name: str) -> str | None:
    """Tiny `shutil.which` wrapper that returns None instead of empty."""
    from shutil import which

    return which(name)


class ShellExecTool(Tool):
    """Run a single command and return its output."""

    name = "shell_exec"
    description = (
        "Run a one-shot command in the workspace shell (PowerShell on Windows, "
        "bash on Linux/macOS). Combines stdout + stderr. Use this for builds, "
        "tests, package management, git, etc. Defaults to a 60s timeout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command line to execute."},
            "timeout": {
                "type": "integer",
                "description": "Maximum seconds before killing the process.",
                "minimum": 1,
                "maximum": 3600,
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory (relative to workspace).",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        command = require_str(arguments, "command")
        timeout = int(arguments.get("timeout") or 60)
        cwd_raw = optional_str(arguments, "cwd")
        cwd = (self.workspace / cwd_raw).resolve() if cwd_raw else self.workspace
        if not cwd.exists() or not cwd.is_dir():
            return error_result(f"cwd does not exist: {cwd}")

        shell_cmd = _default_shell()
        full = [*shell_cmd, command]
        creationflags = 0
        if sys.platform.startswith("win"):
            # Suppress the brief console flash for child processes.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                full,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            return error_result(
                f"Command timed out after {timeout}s: {exc.cmd}\n"
                f"Partial stdout: {exc.stdout or ''!s}\n"
                f"Partial stderr: {exc.stderr or ''!s}"
            )
        except FileNotFoundError as exc:
            return error_result(f"Shell not found: {exc}")
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        body = stdout if not stderr else f"{stdout}\n--- stderr ---\n{stderr}"
        prefix = f"[exit={proc.returncode}] cwd={cwd}\n"
        return ToolResult(
            ok=(proc.returncode == 0),
            output=truncate(prefix + body),
            data={
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cwd": str(cwd),
            },
        )


class ShellSessionTool(Tool):
    """Manage persistent shell sessions via the terminal emulator.

    Subcommands (passed in the `action` argument):
      - `open`: start a session, returns the session id.
      - `send`: send a command line to a session.
      - `read`: read pending output from a session.
      - `close`: terminate a session.
    """

    name = "shell_session"
    description = (
        "Manage a persistent shell session (PowerShell on Windows, bash on "
        "Linux). Use this when a command requires interactive state — e.g. "
        "activating a virtualenv then running pytest — instead of `shell_exec`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "send", "read", "close", "list"],
            },
            "session_id": {"type": "string"},
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
        },
        "required": ["action"],
    }

    def __init__(self, workspace: Path, emulator: TerminalEmulator) -> None:
        self.workspace = workspace
        self.emulator = emulator

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = require_str(arguments, "action")
        if action == "open":
            sid = self.emulator.open_session(self.workspace)
            return text_result(f"Opened shell session {sid}", session_id=sid)
        if action == "list":
            sids = self.emulator.list_sessions()
            return text_result(", ".join(sids) or "(no sessions)", sessions=sids)
        sid = require_str(arguments, "session_id")
        if action == "send":
            cmd = require_str(arguments, "command")
            self.emulator.send(sid, cmd)
            return text_result(f"Sent command to {sid}.")
        if action == "read":
            timeout = float(arguments.get("timeout") or 5)
            out = self.emulator.read(sid, timeout=timeout)
            return text_result(truncate(out) or "(no output)")
        if action == "close":
            self.emulator.close(sid)
            return text_result(f"Closed session {sid}.")
        return error_result(f"Unknown action: {action}")
