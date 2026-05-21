"""Knowledge-embed-once invariant + CLI streaming-flush regression tests.

These cover the two behaviours called out in the user's request:

1.  *"if user uploads any knowledge, make sure its all embedded, to preserve
    token usage so the agent doesnt continuously search knowledge"* — assert
    the agent embeds every note into ``messages[0]`` at session start and
    does NOT insert a ``<retrieved-knowledge>`` block per turn.

2.  *"fix it so it worksxwell"* — the symptom was that ``devin-local run``
    output never appeared when stdout was redirected. Assert the CLI's
    ``_streaming_printer`` actually flushes ``console.file`` on every token.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from devin_local.agent import Agent, AgentConfig
from devin_local.cli import _streaming_printer
from devin_local.inference.backend import ChatChunk
from devin_local.ollama_client import ChatMessage, ChatResponse
from devin_local.tools.base import Tool, text_result
from devin_local.tools.registry import ToolRegistry


class _ScriptedClient:
    def __init__(self, scripted: list[ChatResponse]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, model: str, messages, tools=None, options=None, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": [m.to_dict() for m in messages]})
        if not self._scripted:
            raise AssertionError("ran out of scripted responses")
        return self._scripted.pop(0)

    def summarize(self, *_a, **_kw):  # noqa: ANN001
        return "summary"

    def close(self) -> None:
        pass


@dataclass
class _EchoTool(Tool):
    name: str = "echo"
    description: str = "Echo input."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )

    def run(self, arguments):  # noqa: ANN001
        return text_result(arguments.get("text", ""))


def _resp(content: str = "", tool_calls=None) -> ChatResponse:  # noqa: ANN001
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls or []),
        raw={},
    )


def _build_agent(tmp_path: Path, client, *, embed_all_knowledge: bool = True) -> Agent:
    registry = ToolRegistry.empty()
    registry.register(_EchoTool())
    return Agent(
        AgentConfig(
            model="phi3:mini",
            workspace=tmp_path,
            enable_desktop=False,
            enable_browser=False,
            enable_plugins=False,
            enable_mcp=False,
            enable_knowledge_injection=True,
            enable_skill_injection=False,
            embed_all_knowledge=embed_all_knowledge,
        ),
        client=client,  # type: ignore[arg-type]
        registry=registry,
    )


def test_uploaded_knowledge_is_embedded_in_system_prompt(tmp_path: Path, monkeypatch) -> None:
    """Notes added to the workspace store land verbatim in messages[0]."""
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(user_home))

    client = _ScriptedClient([_resp(content="ok")])
    agent = _build_agent(tmp_path, client)
    agent.knowledge.add("Build", "Run cargo build to compile the rust crate.")
    agent.knowledge.add("Style", "Always run ruff format before committing.")
    agent.initialize()

    system_content = agent.messages[0].content
    assert "## Injected knowledge" in system_content
    assert "Run cargo build to compile the rust crate." in system_content
    assert "Always run ruff format before committing." in system_content


def test_user_level_uploaded_knowledge_embedded(tmp_path: Path, monkeypatch) -> None:
    """User-level (cross-workspace) uploads also land in messages[0]."""
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(user_home))

    client = _ScriptedClient([_resp(content="ok")])
    agent = _build_agent(tmp_path, client)
    agent.user_knowledge.add("GitHub PAT", "Never log or print the GitHub PAT.")
    agent.initialize()

    system_content = agent.messages[0].content
    assert "Never log or print the GitHub PAT." in system_content


def test_embedded_knowledge_is_not_re_searched_per_turn(tmp_path: Path, monkeypatch) -> None:
    """In embed mode, the per-turn `_inject_relevant_context` MUST NOT run.

    Concretely: there is exactly one ``role="system"`` message after several
    user turns (the cached system prompt), not one per turn.
    """
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(user_home))

    client = _ScriptedClient(
        [
            _resp(content="ok 1"),
            _resp(content="ok 2"),
            _resp(content="ok 3"),
        ]
    )
    agent = _build_agent(tmp_path, client)
    agent.knowledge.add("X", "Some workspace note about X.")
    agent.initialize()

    # Three rounds of user→assistant interleaving.
    agent.handle_user("question 1")
    agent.handle_user("question 2")
    agent.handle_user("question 3")

    system_messages = [m for m in agent.messages if m.role == "system"]
    assert len(system_messages) == 1, (
        f"expected exactly one cached system message, got {len(system_messages)} "
        f"— knowledge is being re-injected per turn"
    )


def test_legacy_mode_still_injects_per_turn(tmp_path: Path, monkeypatch) -> None:
    """Setting ``embed_all_knowledge=False`` restores the old TF-IDF behaviour."""
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(user_home))

    client = _ScriptedClient([_resp(content="ok")])
    agent = _build_agent(tmp_path, client, embed_all_knowledge=False)
    agent.knowledge.add("Build", "Run cargo build to compile the rust crate.")
    agent.initialize()
    agent.handle_user("how do i build the rust crate?")

    # Initial cached system prompt + per-turn injected retrieval block = 2.
    system_messages = [m for m in agent.messages if m.role == "system"]
    assert len(system_messages) == 2
    assert any("<retrieved-knowledge>" in m.content for m in system_messages)


def test_system_prompt_cache_invalidates_when_knowledge_changes(
    tmp_path: Path, monkeypatch
) -> None:
    """Adding a note + calling ``refresh_embedded_knowledge`` updates messages[0]."""
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(user_home))

    client = _ScriptedClient(
        [
            _resp(content="ok 1"),
            _resp(content="ok 2"),
        ]
    )
    agent = _build_agent(tmp_path, client)
    agent.initialize()
    before = agent.messages[0].content
    assert "Fresh note about caching" not in before

    agent.knowledge.add("Cache", "Fresh note about caching.")
    agent.refresh_embedded_knowledge()
    after = agent.messages[0].content
    assert "Fresh note about caching." in after
    # Still one system message — not two.
    assert sum(1 for m in agent.messages if m.role == "system") == 1


def test_max_embedded_knowledge_chars_caps_blocks(tmp_path: Path, monkeypatch) -> None:
    """When the corpus exceeds the cap, lower-priority notes drop out cleanly."""
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("DEVIN_LOCAL_HOME", str(user_home))

    client = _ScriptedClient([_resp(content="ok")])
    registry = ToolRegistry.empty()
    registry.register(_EchoTool())
    agent = Agent(
        AgentConfig(
            model="phi3:mini",
            workspace=tmp_path,
            enable_desktop=False,
            enable_browser=False,
            enable_plugins=False,
            enable_mcp=False,
            enable_knowledge_injection=True,
            enable_skill_injection=False,
            embed_all_knowledge=True,
            max_embedded_knowledge_chars=200,
        ),
        client=client,  # type: ignore[arg-type]
        registry=registry,
    )
    agent.knowledge.add("Small", "tiny body")  # fits
    agent.knowledge.add("Huge", "x" * 5000)  # too big; should be dropped
    agent.initialize()
    system_content = agent.messages[0].content
    assert "tiny body" in system_content
    assert "x" * 5000 not in system_content


def test_streaming_printer_flushes_each_delta() -> None:
    """``_streaming_printer`` must call ``console.file.flush()`` on every delta.

    This is the regression fix for the bug where ``devin-local run > log.txt``
    appeared to hang for the entire turn because stdout was buffered.
    """
    fake_console = MagicMock()
    fake_console.file = io.BytesIO()
    fake_console.file.flush = MagicMock()

    emit = _streaming_printer(fake_console)
    for token in ("Hel", "lo, ", "world"):
        emit(ChatChunk(delta=token, done=False))

    assert fake_console.print.call_count == 3
    # Crucially: flush is called once per delta, not just once at end-of-turn.
    assert fake_console.file.flush.call_count == 3


def test_streaming_printer_ignores_done_chunks() -> None:
    """The final ``done=True`` chunk has no delta; printer must skip it."""
    fake_console = MagicMock()
    fake_console.file = io.BytesIO()
    fake_console.file.flush = MagicMock()

    emit = _streaming_printer(fake_console)
    emit(ChatChunk(delta="hi", done=False))
    emit(ChatChunk(delta="", done=True))

    assert fake_console.print.call_count == 1
    assert fake_console.file.flush.call_count == 1
