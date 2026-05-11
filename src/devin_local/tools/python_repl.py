"""Inline Python execution tool.

Useful when the agent needs to do arithmetic, transform structured data, or
prototype a snippet without writing a file. Runs in a fresh subprocess so a
crash doesn't kill the agent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from devin_local.tools.base import Tool, ToolResult, require_str, truncate


class PythonReplTool(Tool):
    """Execute a Python snippet in a fresh subprocess and return stdout."""

    name = "python_exec"
    description = (
        "Run a Python 3 snippet in a fresh subprocess. Useful for quick "
        "calculations or transforming data. The snippet runs with the agent's "
        "workspace as cwd. Output is captured (stdout + stderr)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
        },
        "required": ["code"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        code = require_str(arguments, "code")
        timeout = int(arguments.get("timeout") or 30)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output=f"Python snippet timed out after {timeout}s.")
        body = (proc.stdout or "") + (f"\n--- stderr ---\n{proc.stderr}" if proc.stderr else "")
        return ToolResult(
            ok=(proc.returncode == 0),
            output=truncate(f"[exit={proc.returncode}]\n{body}"),
            data={"exit_code": proc.returncode},
        )
