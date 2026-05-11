"""File-system tools for the agent: read, write, edit, list, grep, find.

These tools are intentionally similar in spirit to the Claude Code / Devin
file primitives so the agent can manipulate a project workspace with simple,
explicit operations. They are sandboxed only by convention — the agent runs
with the user's privileges. We do, however, refuse traversal outside the
configured workspace root by default.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devin_local.tools.base import (
    Tool,
    ToolError,
    ToolResult,
    error_result,
    optional_str,
    require_str,
    text_result,
    truncate,
)


@dataclass
class FileToolContext:
    """Shared state for file tools (currently just the workspace root)."""

    workspace: Path

    def resolve(self, raw_path: str) -> Path:
        """Resolve `raw_path` against the workspace, blocking traversal escapes.

        If `raw_path` is absolute it is honored, but only when it stays inside
        the workspace tree. Relative paths are resolved against the workspace.
        """
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        candidate = candidate.resolve()
        workspace = self.workspace.resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise ToolError(f"Path {candidate} is outside the workspace ({workspace}).") from exc
        return candidate


class ReadFileTool(Tool):
    """Read the contents of a UTF-8 text file."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. Use this before editing "
        "to confirm current contents. Supports an optional line offset/limit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to read (relative or absolute)."},
            "offset": {
                "type": "integer",
                "description": "Optional 1-based starting line.",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Optional max number of lines to return.",
                "minimum": 1,
            },
        },
        "required": ["path"],
    }

    def __init__(self, ctx: FileToolContext) -> None:
        self.ctx = ctx

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.ctx.resolve(require_str(arguments, "path"))
        if not path.exists():
            return error_result(f"File not found: {path}")
        if path.is_dir():
            return error_result(f"Path is a directory, not a file: {path}")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return error_result(f"Could not read {path}: {exc}")

        offset = int(arguments.get("offset") or 1)
        limit = int(arguments.get("limit") or 0)
        lines = text.splitlines()
        if offset > 1 or limit:
            sliced = lines[offset - 1 : (offset - 1 + limit) if limit else None]
        else:
            sliced = lines
        numbered = "\n".join(f"{i + offset:>5}  {line}" for i, line in enumerate(sliced))
        return text_result(truncate(numbered), path=str(path), lines=len(sliced))


class WriteFileTool(Tool):
    """Create or overwrite a file with exact contents."""

    name = "write_file"
    description = (
        "Create or fully overwrite a file. Parent directories are created. "
        "Always read the file first if you intend to modify existing content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, ctx: FileToolContext) -> None:
        self.ctx = ctx

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.ctx.resolve(require_str(arguments, "path"))
        content = optional_str(arguments, "content")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return error_result(f"Could not write {path}: {exc}")
        return text_result(
            f"Wrote {len(content)} bytes to {path}", path=str(path), bytes=len(content)
        )


class EditFileTool(Tool):
    """Replace one exact substring in a file."""

    name = "edit_file"
    description = (
        "Replace the first occurrence of an exact string in a file. "
        "Use this for surgical edits. The `old_string` must match exactly "
        "(including whitespace) and must occur exactly once unless `all` is true."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "all": {"type": "boolean", "description": "Replace all occurrences."},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, ctx: FileToolContext) -> None:
        self.ctx = ctx

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.ctx.resolve(require_str(arguments, "path"))
        old = require_str(arguments, "old_string")
        new = optional_str(arguments, "new_string")
        replace_all = bool(arguments.get("all", False))
        if not path.exists():
            return error_result(f"File not found: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return error_result(f"Could not read {path}: {exc}")
        count = text.count(old)
        if count == 0:
            return error_result(f"Did not find old_string in {path}.")
        if count > 1 and not replace_all:
            return error_result(
                f"old_string occurs {count} times in {path}; pass `all=true` or "
                "provide more surrounding context."
            )
        updated = text.replace(old, new, -1 if replace_all else 1)
        path.write_text(updated, encoding="utf-8")
        return text_result(
            f"Replaced {count if replace_all else 1} occurrence(s) in {path}.",
            path=str(path),
            replacements=count if replace_all else 1,
        )


class ListDirTool(Tool):
    """List entries in a directory."""

    name = "list_dir"
    description = "List immediate children of a directory (files and subdirs)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Defaults to '.'."},
        },
    }

    def __init__(self, ctx: FileToolContext) -> None:
        self.ctx = ctx

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raw = optional_str(arguments, "path", ".") or "."
        path = self.ctx.resolve(raw)
        if not path.exists():
            return error_result(f"Directory not found: {path}")
        if not path.is_dir():
            return error_result(f"Not a directory: {path}")
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines: list[str] = []
        for entry in entries:
            kind = "dir " if entry.is_dir() else "file"
            try:
                size = entry.stat().st_size if entry.is_file() else 0
            except OSError:
                size = 0
            lines.append(f"{kind}  {size:>10}  {entry.name}")
        return text_result("\n".join(lines) or "(empty)", path=str(path), count=len(entries))


class FindFilesTool(Tool):
    """Find files by glob pattern recursively."""

    name = "find_files"
    description = (
        "Find files by glob pattern (e.g. '**/*.py'). Returns paths relative "
        "to the workspace, capped at 200 results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "root": {"type": "string", "description": "Starting directory; defaults to '.'."},
        },
        "required": ["pattern"],
    }

    def __init__(self, ctx: FileToolContext) -> None:
        self.ctx = ctx

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = require_str(arguments, "pattern")
        root = self.ctx.resolve(optional_str(arguments, "root", ".") or ".")
        if not root.is_dir():
            return error_result(f"Root is not a directory: {root}")
        matches: list[str] = []
        for current_root, _dirs, files in os.walk(root):
            for f in files:
                candidate = Path(current_root) / f
                rel = candidate.relative_to(self.ctx.workspace.resolve())
                if fnmatch.fnmatch(str(rel).replace(os.sep, "/"), pattern) or fnmatch.fnmatch(
                    f, pattern
                ):
                    matches.append(str(rel).replace(os.sep, "/"))
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break
        return text_result(
            "\n".join(matches) or "(no matches)", count=len(matches), pattern=pattern
        )


class GrepTool(Tool):
    """Search file contents with a regular expression."""

    name = "grep"
    description = (
        "Search files for a regex pattern. Returns up to 200 matches with "
        "path, line number, and matched line."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "root": {"type": "string", "description": "Starting directory; defaults to '.'."},
            "glob": {
                "type": "string",
                "description": "Optional filename glob filter (e.g. '*.py').",
            },
            "case_insensitive": {"type": "boolean"},
        },
        "required": ["pattern"],
    }

    def __init__(self, ctx: FileToolContext) -> None:
        self.ctx = ctx

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = require_str(arguments, "pattern")
        root = self.ctx.resolve(optional_str(arguments, "root", ".") or ".")
        glob = optional_str(arguments, "glob")
        flags = re.IGNORECASE if arguments.get("case_insensitive") else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return error_result(f"Invalid regex: {exc}")
        results: list[str] = []
        for current_root, _dirs, files in os.walk(root):
            for f in files:
                if glob and not fnmatch.fnmatch(f, glob):
                    continue
                file_path = Path(current_root) / f
                try:
                    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
                        for lineno, line in enumerate(fh, start=1):
                            if regex.search(line):
                                rel = file_path.relative_to(self.ctx.workspace.resolve())
                                results.append(f"{rel}:{lineno}: {line.rstrip()}")
                                if len(results) >= 200:
                                    break
                except OSError:
                    continue
                if len(results) >= 200:
                    break
            if len(results) >= 200:
                break
        return text_result(
            "\n".join(results) or "(no matches)", count=len(results), pattern=pattern
        )
