"""Tests for context compression."""

from __future__ import annotations

from devin_local.context_manager import (
    CompactionConfig,
    ContextManager,
    estimate_messages_tokens,
    estimate_tokens,
)
from devin_local.ollama_client import ChatMessage


def test_estimate_tokens_grows_with_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hi") >= 1
    assert estimate_tokens("hello world") > estimate_tokens("hi")


def test_does_not_compact_when_under_threshold():
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
    ]
    mgr = ContextManager(
        model="llama3.1:8b",
        summarize=lambda text, n: "SUMMARY",
    )
    assert mgr.maybe_compact(messages) == messages


def test_compacts_when_over_threshold():
    # phi3:mini has a 4k window; with 50 long messages we will blow past 70%.
    big = "x" * 4_000
    messages: list[ChatMessage] = [ChatMessage(role="system", content="sys")]
    for i in range(50):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(ChatMessage(role=role, content=f"{role}-{i}: " + big))

    calls: list[str] = []

    def fake_summary(text: str, n: int) -> str:
        calls.append(text)
        return "compacted notes"

    mgr = ContextManager(
        model="phi3:mini",
        summarize=fake_summary,
        config=CompactionConfig(threshold_ratio=0.7, keep_tail=4),
    )
    compacted = mgr.maybe_compact(messages)
    # System preserved, then one summary, then tail.
    assert compacted[0].role == "system"
    assert "compaction-summary" in compacted[1].content
    assert len(compacted) == 1 + 1 + 4
    assert calls, "summarize() must have been called"
    assert estimate_messages_tokens(compacted) < estimate_messages_tokens(messages)


def test_compaction_skipped_when_history_short():
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hi"),
    ]
    mgr = ContextManager(model="phi3:mini", summarize=lambda text, n: "X")
    assert mgr.maybe_compact(messages) == messages
