"""Tests for the file tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from devin_local.tools.base import ToolError
from devin_local.tools.file_tools import (
    EditFileTool,
    FileToolContext,
    FindFilesTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)


@pytest.fixture
def ctx(tmp_path: Path) -> FileToolContext:
    return FileToolContext(workspace=tmp_path)


def test_write_then_read_roundtrip(ctx: FileToolContext, tmp_path: Path):
    writer = WriteFileTool(ctx)
    reader = ReadFileTool(ctx)

    write_result = writer.run({"path": "hello.txt", "content": "hi there\nline two\n"})
    assert write_result.ok
    assert (tmp_path / "hello.txt").read_text() == "hi there\nline two\n"

    read_result = reader.run({"path": "hello.txt"})
    assert read_result.ok
    assert "hi there" in read_result.output
    # File output is numbered.
    assert "1  hi there" in read_result.output


def test_edit_replace_unique(ctx: FileToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("alpha beta gamma")
    editor = EditFileTool(ctx)
    res = editor.run({"path": "f.txt", "old_string": "beta", "new_string": "DELTA"})
    assert res.ok
    assert (tmp_path / "f.txt").read_text() == "alpha DELTA gamma"


def test_edit_refuses_non_unique_without_all(ctx: FileToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("x x x")
    editor = EditFileTool(ctx)
    res = editor.run({"path": "f.txt", "old_string": "x", "new_string": "y"})
    assert not res.ok
    assert "occurs 3 times" in res.output


def test_edit_all_replaces_all(ctx: FileToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("x x x")
    editor = EditFileTool(ctx)
    res = editor.run({"path": "f.txt", "old_string": "x", "new_string": "y", "all": True})
    assert res.ok
    assert (tmp_path / "f.txt").read_text() == "y y y"


def test_list_dir(ctx: FileToolContext, tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    res = ListDirTool(ctx).run({"path": "."})
    assert res.ok
    assert "a.txt" in res.output
    assert "sub" in res.output


def test_find_files_pattern(ctx: FileToolContext, tmp_path: Path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("b")
    (tmp_path / "c.txt").write_text("c")
    res = FindFilesTool(ctx).run({"pattern": "*.py"})
    assert res.ok
    assert "a.py" in res.output
    assert "b.py" in res.output
    assert "c.txt" not in res.output


def test_grep(ctx: FileToolContext, tmp_path: Path):
    (tmp_path / "a.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "b.py").write_text("def goodbye():\n    return 0\n")
    res = GrepTool(ctx).run({"pattern": r"def \w+"})
    assert res.ok
    assert "a.py:1" in res.output
    assert "b.py:1" in res.output


def test_workspace_escape_blocked(ctx: FileToolContext, tmp_path: Path):
    """Tool path resolution must refuse to escape the workspace.

    Tools raise `ToolError` for invalid arguments — the dispatcher converts
    that into a failed `ToolResult` (see `test_tool_registry` for the
    dispatched path).
    """
    reader = ReadFileTool(ctx)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope")
    with pytest.raises(ToolError):
        reader.run({"path": str(outside)})


def test_workspace_escape_blocked_via_dispatch(tmp_path: Path):
    """The dispatched path should surface the same condition as `ok=False`."""
    from devin_local.tools.registry import ToolRegistry

    ctx = FileToolContext(workspace=tmp_path)
    reg = ToolRegistry.empty()
    reg.register(ReadFileTool(ctx))
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope")
    res = reg.dispatch("read_file", {"path": str(outside)})
    assert not res.ok
    assert "outside the workspace" in res.output


def test_resolve_traversal_raises_tool_error(ctx: FileToolContext):
    with pytest.raises(ToolError):
        FileToolContext(workspace=ctx.workspace).resolve("../outside")
