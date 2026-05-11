"""End-to-end Hello-World task using the real toolbelt + scripted Ollama.

This is the verification scenario from the README, exercised against the
real `ToolRegistry` (file tools + shell exec) with a deterministic fake
Ollama client. Demonstrates that:

    user → write_file → shell_exec(python hello.py) → final message

works end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

from devin_local.agent import Agent, AgentConfig
from devin_local.ollama_client import ChatMessage, ChatResponse


def _scripted(messages: list[ChatResponse]):  # noqa: ANN202
    class Client:
        def __init__(self) -> None:
            self._q = list(messages)
            self.calls: list[dict] = []

        def chat(self, *, model, messages, tools=None, options=None, **kw):  # noqa: ANN001
            self.calls.append({"messages": [m.to_dict() for m in messages]})
            if not self._q:
                raise AssertionError("scripted client ran out")
            return self._q.pop(0)

        def summarize(self, *a, **kw):  # noqa: ANN001
            return ""

        def close(self):
            return None

    return Client()


def _resp(content: str = "", tool_calls=None):  # noqa: ANN001, ANN202
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls or []),
        raw={},
    )


def test_hello_world_round_trip(tmp_path: Path):
    """The agent writes hello.py, runs it, sees 'Hello, world!', and reports."""
    workspace = tmp_path
    client = _scripted(
        [
            _resp(
                tool_calls=[
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {
                                "path": "hello.py",
                                "content": "print('Hello, world!')\n",
                            },
                        }
                    }
                ]
            ),
            _resp(
                tool_calls=[
                    {
                        "function": {
                            "name": "shell_exec",
                            "arguments": {
                                "command": f"{sys.executable} hello.py",
                                "timeout": 30,
                            },
                        }
                    }
                ]
            ),
            _resp(content=("Created hello.py and ran it. stdout was: Hello, world!")),
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
    )
    turn = agent.handle_user("Create hello.py that prints 'Hello, world!', then run it.")
    assert turn.iterations == 3
    assert "Hello, world!" in turn.assistant_text
    # The file must actually exist on disk.
    assert (workspace / "hello.py").exists()
    assert "Hello, world!" in (workspace / "hello.py").read_text()
    # Among tool results: the second one is the shell exec and must have ok=True.
    names = [name for name, _result in turn.tool_results]
    assert names == ["write_file", "shell_exec"]
    assert turn.tool_results[1][1].ok, "python hello.py must have exited 0"
    assert "Hello, world!" in turn.tool_results[1][1].output
