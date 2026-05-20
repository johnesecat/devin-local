"""Tools-schema pruning tests.

The Agent ships every registered tool's JSON-Schema to the model on every
turn. On CPU-only hardware with 10+ tools registered, that schema can be
3-4 KB of prefill tokens before the model even starts generating. Smart
mode trims it to a relevant subset based on keyword overlap with the
latest user message, while always keeping core file/shell tools.
"""

from __future__ import annotations

from pathlib import Path

from devin_local.agent import Agent, AgentConfig, _tokenize_for_match
from devin_local.inference.backend import (
    ChatResponse,
    InferenceBackend,
)
from devin_local.inference.types import ChatMessage


class _NoopBackend(InferenceBackend):
    def chat(self, *args, **kwargs):  # noqa: ANN001, ANN201
        return ChatResponse(message=ChatMessage(role="assistant", content=""))

    def summarize(self, *args, **kwargs) -> str:  # noqa: ANN001
        return ""

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return []


def _agent(tmp_path: Path, **overrides) -> Agent:
    config = AgentConfig(
        workspace=tmp_path,
        model="llama3.2:1b",
        backend="ollama",
        max_iterations=1,
        **overrides,
    )
    return Agent(config, backend=_NoopBackend())


def _schema(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_tokenize_for_match_drops_punctuation_and_lowercases() -> None:
    assert _tokenize_for_match("Open the BROWSER and click!") == [
        "open",
        "the",
        "browser",
        "and",
        "click",
    ]
    assert _tokenize_for_match("") == []


def test_full_mode_returns_all_schemas_unchanged(tmp_path: Path) -> None:
    agent = _agent(tmp_path, tools_schema_mode="full")
    agent.messages = [ChatMessage(role="user", content="open a browser tab")]
    schemas = [
        _schema("read_file"),
        _schema("write_file"),
        _schema("shell_exec"),
        _schema("browser_navigate"),
        _schema("desktop_click"),
        _schema("python_repl"),
        _schema("knowledge_search"),
        _schema("knowledge_read"),
    ]
    result = agent._select_relevant_tools(schemas)
    assert result == schemas


def test_smart_mode_with_no_user_message_returns_all(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.messages = []
    schemas = [_schema(f"tool_{i}") for i in range(10)]
    assert agent._select_relevant_tools(schemas) == schemas


def test_smart_mode_keeps_always_include_tools_even_without_keyword_match(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    agent.messages = [
        ChatMessage(role="user", content="screenshot the whole desktop please"),
    ]
    schemas = [
        _schema("read_file", "Read the contents of a file."),
        _schema("write_file", "Write content to a file."),
        _schema("edit_file", "Edit a file in place."),
        _schema("list_directory", "List a directory."),
        _schema("shell_exec", "Run a shell command."),
        _schema("knowledge_search", "Search the knowledge index."),
        _schema("knowledge_read", "Read a knowledge note."),
        _schema("desktop_screenshot", "Take a screenshot of the desktop."),
    ]
    chosen = agent._select_relevant_tools(schemas)
    chosen_names = {s["function"]["name"] for s in chosen}
    # Always-include set must remain even when no keyword match.
    for name in (
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "shell_exec",
        "knowledge_search",
        "knowledge_read",
    ):
        assert name in chosen_names, f"always-include {name!r} got pruned"
    # Keyword match: 'screenshot' / 'desktop' should pull in desktop_screenshot.
    assert "desktop_screenshot" in chosen_names


def test_smart_mode_drops_tools_unrelated_to_the_request(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.messages = [
        ChatMessage(role="user", content="please write a python hello world to hello.py"),
    ]
    schemas = [
        _schema("read_file"),
        _schema("write_file"),
        _schema("edit_file"),
        _schema("list_directory"),
        _schema("shell_exec"),
        _schema("knowledge_search"),
        _schema("knowledge_read"),
        _schema("browser_navigate", "Navigate the embedded browser to a URL."),
        _schema("desktop_screenshot", "Take a screenshot of the desktop."),
        _schema("desktop_click", "Click somewhere on the desktop."),
    ]
    chosen = agent._select_relevant_tools(schemas)
    chosen_names = {s["function"]["name"] for s in chosen}
    # Off-topic tools should be pruned out of the shipped subset.
    assert "browser_navigate" not in chosen_names
    assert "desktop_screenshot" not in chosen_names
    assert "desktop_click" not in chosen_names
    # Strictly fewer schemas shipped vs the full toolbelt.
    assert len(chosen) < len(schemas)


def test_smart_mode_falls_back_to_all_when_nothing_matches(tmp_path: Path) -> None:
    """If the smart filter would return an empty list (e.g. all tools are
    off-topic AND there are no always-include matches), the agent must
    return the full set rather than ship zero tools."""
    agent = _agent(tmp_path, always_include_tools=())
    agent.messages = [ChatMessage(role="user", content="xyzzy plugh frobnicate")]
    schemas = [
        _schema("read_file", "Read the contents of a file."),
        _schema("write_file", "Write content to a file."),
        _schema("shell_exec", "Run a shell command."),
        _schema("browser_navigate", "Navigate the embedded browser."),
        _schema("desktop_click", "Click on the desktop."),
        _schema("python_repl", "Run python code."),
    ]
    chosen = agent._select_relevant_tools(schemas)
    assert chosen == schemas


def test_smart_mode_skips_pruning_with_fewer_than_six_tools(tmp_path: Path) -> None:
    """Pruning adds branching cost; for tiny registries it's not worth it."""
    agent = _agent(tmp_path)
    agent.messages = [ChatMessage(role="user", content="open browser")]
    schemas = [_schema("a"), _schema("b"), _schema("c"), _schema("d"), _schema("e")]
    assert agent._select_relevant_tools(schemas) == schemas
