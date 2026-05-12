"""Tests for the ChatML chat template used by layered / HF backends."""

from __future__ import annotations

import json

from devin_local.inference.chat_template import (
    format_chatml_prompt,
    render_system_with_tools,
)
from devin_local.inference.types import ChatMessage


def test_render_system_with_tools_appends_tool_catalog() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "write_file", "description": "...", "parameters": {}},
        }
    ]
    rendered = render_system_with_tools("Be helpful.", tools)
    assert rendered.startswith("Be helpful.")
    assert "<tools>" in rendered
    assert "write_file" in rendered
    # No trailing assistant role yet
    assert "<|im_start|>" not in rendered


def test_render_system_with_tools_no_tools_is_passthrough() -> None:
    assert render_system_with_tools("hi", None) == "hi"
    assert render_system_with_tools("hi", []) == "hi"


def test_format_chatml_prompt_has_required_segments() -> None:
    msgs = [
        ChatMessage(role="system", content="You are devin-local."),
        ChatMessage(role="user", content="hi"),
    ]
    prompt = format_chatml_prompt(msgs)
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_format_chatml_prompt_injects_system_when_missing_but_tools_given() -> None:
    msgs = [ChatMessage(role="user", content="hi")]
    tools = [
        {
            "type": "function",
            "function": {"name": "shell_exec", "description": "...", "parameters": {}},
        }
    ]
    prompt = format_chatml_prompt(msgs, tools=tools)
    # An auto-injected system message should appear first.
    assert prompt.startswith("<|im_start|>system\n")
    assert "shell_exec" in prompt


def test_format_chatml_prompt_renders_tool_results_as_observations() -> None:
    msgs = [
        ChatMessage(role="user", content="run it"),
        ChatMessage(
            role="tool",
            name="shell_exec",
            content=json.dumps({"ok": True, "stdout": "done"}),
        ),
    ]
    prompt = format_chatml_prompt(msgs)
    assert '<observation tool="shell_exec">' in prompt
    assert "stdout" in prompt


def test_format_chatml_prompt_replays_assistant_tool_calls() -> None:
    msgs = [
        ChatMessage(role="user", content="run it"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"function": {"name": "shell_exec", "arguments": {"command": "ls"}}}],
        ),
    ]
    prompt = format_chatml_prompt(msgs)
    assert "<tool_call>" in prompt
    assert "shell_exec" in prompt
