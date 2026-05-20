"""Tools that let the agent query the per-session knowledge index on demand.

The agent's system prompt only embeds a tiny manifest (relative paths +
1-line summaries) for the configured knowledge directory. Full file bodies
are loaded ONLY when the agent calls these tools, so the operator pays
zero token cost for unused knowledge.

Two tools:

- ``knowledge_search`` — TF-IDF over the index, returns top-K snippets.
- ``knowledge_read`` — load a specific file by its relative path.
"""

from __future__ import annotations

from typing import Any

from devin_local.knowledge.index import KnowledgeIndex
from devin_local.tools.base import (
    Tool,
    ToolResult,
    error_result,
    require_str,
    text_result,
    truncate,
)

_MAX_OUTPUT_CHARS = 12_000


class KnowledgeSearchTool(Tool):
    """Top-K TF-IDF search over the session's knowledge directory."""

    name = "knowledge_search"
    description = (
        "Search the session's knowledge directory for passages relevant to a query. "
        "Returns the top-K matching files with focused excerpts. Prefer this over "
        "reading entire files. The corpus and its manifest are described in your "
        "system prompt under '## Knowledge directory'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search query (keywords, identifiers, error messages).",
            },
            "k": {
                "type": "integer",
                "description": "Maximum number of hits to return. Default 4.",
                "minimum": 1,
                "maximum": 12,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, index: KnowledgeIndex) -> None:
        self._index = index

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = require_str(arguments, "query")
        try:
            k_arg = arguments.get("k", 4)
            k = int(k_arg) if k_arg is not None else 4
        except (TypeError, ValueError):
            k = 4
        k = max(1, min(12, k))
        try:
            hits = self._index.search(query, k=k)
        except Exception as exc:  # noqa: BLE001
            return error_result(f"knowledge_search failed: {exc!r}")
        if not hits:
            return text_result(
                f"No matches for {query!r} in {self._index.root}.",
                root=str(self._index.root),
                matches=0,
            )
        rendered = "\n\n---\n\n".join(h.to_block() for h in hits)
        return text_result(
            truncate(rendered, _MAX_OUTPUT_CHARS),
            matches=len(hits),
            paths=[h.relpath for h in hits],
        )


class KnowledgeReadTool(Tool):
    """Load the full body of a knowledge file by relative path."""

    name = "knowledge_read"
    description = (
        "Read a specific knowledge file by its relative path within the session's "
        "knowledge directory. Paths come from `knowledge_search` results or the "
        "manifest in the system prompt. Path traversal outside the root is refused."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Relative path within the knowledge directory (forward slashes). "
                    "Example: 'guides/install.md'."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, index: KnowledgeIndex) -> None:
        self._index = index

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = require_str(arguments, "path")
        try:
            body = self._index.read(path)
        except FileNotFoundError as exc:
            return error_result(str(exc))
        except ValueError as exc:
            return error_result(str(exc))
        except Exception as exc:  # noqa: BLE001
            return error_result(f"knowledge_read failed: {exc!r}")
        return text_result(
            truncate(body, _MAX_OUTPUT_CHARS),
            path=path,
            bytes=len(body),
        )


class KnowledgeListTool(Tool):
    """List every knowledge file in the session's knowledge directory."""

    name = "knowledge_list"
    description = (
        "List every file in the session's knowledge directory with its title and "
        "1-line summary. Use this when `knowledge_search` returns no hits to make "
        "sure you understand what is available."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, index: KnowledgeIndex) -> None:
        self._index = index

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        if not self._index.notes:
            return text_result(
                f"(empty) No knowledge files under {self._index.root}.",
                root=str(self._index.root),
                count=0,
            )
        manifest = self._index.manifest(max_entries=500)
        return text_result(
            truncate(manifest, _MAX_OUTPUT_CHARS),
            count=len(self._index.notes),
        )
