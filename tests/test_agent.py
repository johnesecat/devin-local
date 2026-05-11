"""End-to-end tests for the Agent harness, with a fake Ollama client.

We don't require a running Ollama daemon for these tests; we plug a fake
client implementing just the `chat()` / `summarize()` / `close()` surface
the agent uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devin_local.agent import Agent, AgentConfig
from devin_local.ollama_client import ChatMessage, ChatResponse
from devin_local.tools.base import Tool, text_result
from devin_local.tools.registry import ToolRegistry


class _ScriptedClient:
    """Fake OllamaClient that replays a pre-recorded list of ChatResponses."""

    def __init__(self, scripted: list[ChatResponse]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, model: str, messages, tools=None, options=None, **kwargs):  # noqa: ANN001
        self.calls.append(
            {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "tools": tools,
                "options": options,
            }
        )
        if not self._scripted:
            raise AssertionError("Scripted client ran out of responses")
        return self._scripted.pop(0)

    def summarize(self, *_a: Any, **_kw: Any) -> str:  # pragma: no cover - unused here
        return "summary"

    def close(self) -> None:
        pass


@dataclass
class _NoteTool(Tool):
    name: str = "make_note"
    description: str = "Write a note to a file."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )

    workspace: Path | None = None

    def run(self, arguments):  # noqa: ANN001
        assert self.workspace is not None
        out = self.workspace / "note.txt"
        out.write_text(arguments["text"], encoding="utf-8")
        return text_result(f"wrote {out}")


def _resp(content: str = "", tool_calls=None) -> ChatResponse:  # noqa: ANN001
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls or []),
        raw={},
    )


def test_agent_runs_tool_and_returns_final_message(tmp_path: Path):
    workspace = tmp_path
    registry = ToolRegistry.empty()
    note_tool = _NoteTool(workspace=workspace)
    registry.register(note_tool)

    client = _ScriptedClient(
        [
            _resp(
                tool_calls=[
                    {
                        "function": {
                            "name": "make_note",
                            "arguments": {"text": "hello world"},
                        }
                    }
                ]
            ),
            _resp(content="Done. I wrote the note."),
        ]
    )
    agent = Agent(
        AgentConfig(
            model="phi3:mini",
            workspace=workspace,
            enable_desktop=False,
            enable_browser=False,
            enable_plugins=False,
            enable_mcp=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
        ),
        client=client,  # type: ignore[arg-type]
        registry=registry,
    )

    turn = agent.handle_user("create a note that says hello world")
    assert turn.iterations == 2
    assert "Done" in turn.assistant_text
    assert (workspace / "note.txt").read_text() == "hello world"
    assert agent.messages[0].role == "system"
    # One tool result message must be present.
    assert any(m.role == "tool" for m in agent.messages)


def test_agent_handles_unknown_tool_call_gracefully(tmp_path: Path):
    registry = ToolRegistry.empty()
    client = _ScriptedClient(
        [
            _resp(tool_calls=[{"function": {"name": "does_not_exist", "arguments": {}}}]),
            _resp(content="Could not call that tool."),
        ]
    )
    agent = Agent(
        AgentConfig(
            model="phi3:mini",
            workspace=tmp_path,
            enable_desktop=False,
            enable_browser=False,
            enable_plugins=False,
            enable_mcp=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
        ),
        client=client,  # type: ignore[arg-type]
        registry=registry,
    )
    turn = agent.handle_user("hi")
    assert "Could not call" in turn.assistant_text


def test_agent_stops_at_max_iterations(tmp_path: Path):
    registry = ToolRegistry.empty()

    # Always returns another tool call so the agent never settles.
    class LoopClient:
        def chat(self, **_kw):  # noqa: ANN001
            return _resp(tool_calls=[{"function": {"name": "ghost", "arguments": {}}}])

        def summarize(self, *_a, **_kw):  # noqa: ANN001
            return "x"

        def close(self):
            pass

    agent = Agent(
        AgentConfig(
            model="phi3:mini",
            workspace=tmp_path,
            max_iterations=3,
            enable_desktop=False,
            enable_browser=False,
            enable_plugins=False,
            enable_mcp=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
        ),
        client=LoopClient(),  # type: ignore[arg-type]
        registry=registry,
    )
    turn = agent.handle_user("go")
    assert turn.iterations == 3
    assert "max iterations" in turn.assistant_text
