"""Directory-backed knowledge index + tools."""

from __future__ import annotations

import pathlib

from devin_local.knowledge.index import KnowledgeIndex
from devin_local.tools.knowledge_tools import (
    KnowledgeListTool,
    KnowledgeReadTool,
    KnowledgeSearchTool,
)


def _populate(root: pathlib.Path) -> None:
    (root / "guides").mkdir(parents=True, exist_ok=True)
    (root / "guides" / "install.md").write_text(
        "# Install\nRun `pip install devin-local`. Needs Python 3.10+.",
        encoding="utf-8",
    )
    (root / "guides" / "tools.md").write_text(
        "# Tool primer\nThe shell_exec tool runs commands. read_file reads files.",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("Project shipped on 2025-01-01.", encoding="utf-8")
    (root / ".hidden.md").write_text("ignored", encoding="utf-8")


def test_index_reload_lists_supported_files(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    rels = {n.relpath for n in idx.notes}
    assert "guides/install.md" in rels
    assert "guides/tools.md" in rels
    assert "notes.txt" in rels


def test_manifest_includes_titles_and_excerpts(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    manifest = idx.manifest(max_entries=10)
    assert "Knowledge directory" in manifest
    assert "guides/install.md" in manifest
    assert "guides/tools.md" in manifest
    assert "Install" in manifest


def test_search_returns_relevant_hits(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    hits = idx.search("shell_exec", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].relpath.endswith("tools.md")


def test_search_empty_corpus_returns_empty(tmp_path):
    idx = KnowledgeIndex.open(tmp_path)
    assert idx.search("anything", k=2) == []


def test_read_rejects_path_traversal(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    import pytest

    with pytest.raises(ValueError):
        idx.read("../outside.md")


def test_knowledge_search_tool_returns_hits_payload(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    tool = KnowledgeSearchTool(idx)
    result = tool.run({"query": "pip install"})
    assert result.ok is True
    assert "install.md" in result.output


def test_knowledge_read_tool_reads_file(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    tool = KnowledgeReadTool(idx)
    result = tool.run({"path": "guides/install.md"})
    assert result.ok is True
    assert "pip install devin-local" in result.output


def test_knowledge_read_tool_rejects_traversal(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    tool = KnowledgeReadTool(idx)
    result = tool.run({"path": "../escape.md"})
    assert result.ok is False


def test_knowledge_list_tool_lists_files(tmp_path):
    _populate(tmp_path)
    idx = KnowledgeIndex.open(tmp_path)
    tool = KnowledgeListTool(idx)
    result = tool.run({})
    assert result.ok is True
    assert "guides/install.md" in result.output
    assert "notes.txt" in result.output
