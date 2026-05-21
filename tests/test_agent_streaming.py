"""End-to-end test for Agent.handle_user(stream=True) using a scripted backend."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from devin_local.agent import Agent, AgentConfig
from devin_local.inference.backend import ChatChunk, ChatResponse, InferenceBackend
from devin_local.inference.types import ChatMessage


class _ScriptedBackend(InferenceBackend):
    """Yields a predetermined sequence of chunks per call.

    Each entry in `script` is a list of (delta, done, optional_message)
    chunks for one call to chat()/stream().
    """

    name = "scripted"

    def __init__(self, script: list[list[tuple[str, bool, ChatMessage | None]]]) -> None:
        self._script = list(script)
        self.calls = 0

    def chat(self, model, messages, tools=None, options=None, keep_alive=None) -> ChatResponse:
        chunks = list(self.stream(model, messages, tools, options, keep_alive))
        final = next(c for c in chunks if c.done)
        assert final.message is not None
        return ChatResponse(message=final.message, done=True)

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> Iterator[ChatChunk]:
        if self.calls >= len(self._script):
            raise AssertionError("scripted backend exhausted")
        chunks = self._script[self.calls]
        self.calls += 1
        for delta, done, message in chunks:
            yield ChatChunk(delta=delta, done=done, message=message)

    def summarize(self, model, text, max_tokens=512, instruction=None) -> str:
        return text[: max_tokens // 2]

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["scripted"]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_agent_streams_token_deltas_to_observers(workspace: Path) -> None:
    final_msg = ChatMessage(role="assistant", content="Hello, world!", tool_calls=[])
    backend = _ScriptedBackend(
        script=[
            [
                ("Hello, ", False, None),
                ("world!", False, None),
                ("", True, final_msg),
            ]
        ]
    )
    agent = Agent(
        AgentConfig(
            workspace=workspace,
            model="any",
            enable_mcp=False,
            enable_plugins=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
        ),
        backend=backend,
    )

    deltas: list[str] = []

    def _capture(chunk: ChatChunk) -> None:
        if not chunk.done and chunk.delta:
            deltas.append(chunk.delta)

    agent.add_stream_observer(_capture)
    try:
        turn = agent.handle_user("hi", stream=True)
    finally:
        agent.shutdown()
    assert deltas == ["Hello, ", "world!"]
    assert turn.assistant_text == "Hello, world!"
    assert turn.iterations == 1


def test_agent_streaming_runs_tool_calls(workspace: Path) -> None:
    # Turn 1: model wants to write a file.
    turn1_final = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "out.txt", "content": "hi"},
                }
            }
        ],
    )
    # Turn 2: model confirms.
    turn2_final = ChatMessage(role="assistant", content="Done.", tool_calls=[])
    backend = _ScriptedBackend(
        script=[
            [("", True, turn1_final)],
            [("Done.", False, None), ("", True, turn2_final)],
        ]
    )
    agent = Agent(
        AgentConfig(
            workspace=workspace,
            model="any",
            enable_mcp=False,
            enable_plugins=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
            enable_desktop=False,
            enable_browser=False,
        ),
        backend=backend,
    )
    try:
        turn = agent.handle_user("write out.txt with 'hi'", stream=True)
    finally:
        agent.shutdown()
    assert turn.iterations == 2
    assert turn.assistant_text == "Done."
    # File was actually written by the real WriteFileTool.
    assert (workspace / "out.txt").read_text() == "hi"


def test_stream_user_yields_chunks_before_turn_completes(workspace: Path) -> None:
    """stream_user() must yield chunks incrementally, not buffer them all
    until the agent turn finishes (Devin Review BUG_0002).

    Adversarial design: the backend yields chunk 0, then blocks on a
    threading.Event. The Event is only released by the consumer AFTER it
    receives chunk 0 from the generator. If stream_user buffered chunks
    until handle_user returns, the consumer would never receive chunk 0,
    the Event would never be set, the backend's `wait(5.0)` would return
    False and raise AssertionError inside the agent loop, and the test
    would fail. The test passing implies incremental delivery.
    """

    import threading
    from collections.abc import Iterator as _Iter

    release_next = threading.Event()

    class _BlockingBackend(InferenceBackend):
        """Streams 3 chunks; blocks on an Event between the 1st and 2nd."""

        name = "blocking"

        def chat(self, model, messages, tools=None, options=None, keep_alive=None):
            raise NotImplementedError

        def stream(
            self,
            model: str,
            messages: list[ChatMessage],
            tools=None,
            options=None,
            keep_alive=None,
        ) -> _Iter[ChatChunk]:
            yield ChatChunk(delta="first ", done=False, message=None)
            assert release_next.wait(5.0), (
                "consumer never received first chunk before backend's 2nd yield"
            )
            yield ChatChunk(delta="second ", done=False, message=None)
            yield ChatChunk(
                delta="",
                done=True,
                message=ChatMessage(role="assistant", content="first second "),
            )

        def summarize(self, model, text, max_tokens=512, instruction=None) -> str:
            return text

        def is_available(self) -> bool:
            return True

        def list_models(self) -> list[str]:
            return ["blocking"]

    backend = _BlockingBackend()
    agent = Agent(
        AgentConfig(
            workspace=workspace,
            model="any",
            enable_mcp=False,
            enable_plugins=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
        ),
        backend=backend,
    )
    try:
        deltas: list[str] = []
        for chunk in agent.stream_user("go"):
            if not chunk.done and chunk.delta:
                deltas.append(chunk.delta)
            if chunk.delta == "first ":
                release_next.set()
        assert deltas == ["first ", "second "]
    finally:
        release_next.set()
        agent.shutdown()


def test_tool_started_fires_before_tool_executes(workspace: Path) -> None:
    """tool_started observers MUST fire before the tool is dispatched, not after
    (Devin Review BUG_0001).

    Adversarial design: the tool_started observer asserts that the file the
    write_file tool is about to create does NOT yet exist on disk. If
    tool_started fires after dispatch (the bug), the file will already exist
    when the observer runs and the assertion will fail. The test passing
    implies the observer fires before tool execution.
    """
    turn1_final = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "before_or_after.txt", "content": "hi"},
                }
            }
        ],
    )
    turn2_final = ChatMessage(role="assistant", content="Done.", tool_calls=[])
    backend = _ScriptedBackend(
        script=[
            [("", True, turn1_final)],
            [("", True, turn2_final)],
        ]
    )
    agent = Agent(
        AgentConfig(
            workspace=workspace,
            model="any",
            enable_mcp=False,
            enable_plugins=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
            enable_desktop=False,
            enable_browser=False,
        ),
        backend=backend,
    )

    target = workspace / "before_or_after.txt"
    events: list[tuple[str, bool]] = []  # (event_kind, file_exists_at_emit)

    def _on_start(name: str, args: dict[str, Any]) -> None:
        events.append(("started", target.exists()))

    def _on_finish(name: str, args: dict[str, Any], result: Any) -> None:
        events.append(("finished", target.exists()))

    agent.add_tool_start_observer(_on_start)
    agent.add_tool_observer(_on_finish)
    try:
        turn = agent.handle_user("write the file", stream=False)
    finally:
        agent.shutdown()
    # Both events fired.
    assert [e[0] for e in events] == ["started", "finished"]
    # tool_started observed the file NOT yet existing (i.e. before dispatch).
    assert events[0] == ("started", False), (
        f"tool_started fired AFTER dispatch — file already existed when "
        f"observer ran: events={events}"
    )
    # tool_finished observed the file existing (i.e. after dispatch).
    assert events[1] == ("finished", True)
    # And the actual write happened.
    assert target.read_text() == "hi"
    assert turn.iterations == 2


def test_agent_caches_system_prompt_across_turns(workspace: Path) -> None:
    backend = _ScriptedBackend(
        script=[
            [("ack", False, None), ("", True, ChatMessage(role="assistant", content="ack"))],
            [("ack2", False, None), ("", True, ChatMessage(role="assistant", content="ack2"))],
        ]
    )
    agent = Agent(
        AgentConfig(
            workspace=workspace,
            model="any",
            enable_mcp=False,
            enable_plugins=False,
            enable_knowledge_injection=False,
            enable_skill_injection=False,
            enable_desktop=False,
            enable_browser=False,
        ),
        backend=backend,
    )
    try:
        agent.handle_user("hi", stream=True)
        first_prompt = agent._cached_system_prompt
        first_snapshot = agent._cached_tool_names_snapshot
        agent.handle_user("again", stream=True)
        second_prompt = agent._cached_system_prompt
        second_snapshot = agent._cached_tool_names_snapshot
    finally:
        agent.shutdown()
    assert first_prompt is not None
    # Same object identity => cache hit (we only rebuild when tools change).
    assert first_prompt is second_prompt
    assert first_snapshot == second_snapshot
