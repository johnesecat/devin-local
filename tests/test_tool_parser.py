"""Tests for the inline tool-call parser used by non-Ollama backends."""

from __future__ import annotations

from devin_local.inference.tool_parser import parse_tool_calls, tool_calls_to_wire


def test_parse_no_tool_call_returns_text_unchanged() -> None:
    text, calls = parse_tool_calls("Hello, world!")
    assert text == "Hello, world!"
    assert calls == []


def test_parse_single_tool_call_tag() -> None:
    msg = (
        "Sure, I'll write the file.\n"
        '<tool_call>{"name": "write_file", '
        '"arguments": {"path": "hello.py", "content": "print(1)"}}</tool_call>'
    )
    text, calls = parse_tool_calls(msg)
    assert text.strip() == "Sure, I'll write the file."
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments == {"path": "hello.py", "content": "print(1)"}


def test_parse_multiple_tool_calls_in_one_turn() -> None:
    msg = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
        "\nLet me also check b:\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "b"}}</tool_call>'
    )
    text, calls = parse_tool_calls(msg)
    assert "Let me also check b" in text
    assert len(calls) == 2
    assert [c.arguments["path"] for c in calls] == ["a", "b"]


def test_parse_fenced_json_tool_call() -> None:
    msg = 'Running:\n```tool_call\n{"name": "shell_exec", "arguments": {"command": "ls"}}\n```'
    text, calls = parse_tool_calls(msg)
    assert "Running:" in text
    assert len(calls) == 1
    assert calls[0].name == "shell_exec"


def test_parse_bare_json_tool_call_as_last_resort() -> None:
    msg = '{"name": "list_dir", "arguments": {"path": "."}}'
    text, calls = parse_tool_calls(msg)
    assert text == ""
    assert len(calls) == 1
    assert calls[0].name == "list_dir"


def test_parse_string_arguments_are_decoded() -> None:
    msg = '<tool_call>{"name": "write_file", "arguments": "{\\"path\\": \\"x\\"}"}</tool_call>'
    _, calls = parse_tool_calls(msg)
    assert len(calls) == 1
    assert calls[0].arguments == {"path": "x"}


def test_tool_calls_to_wire_shape_matches_ollama() -> None:
    msg = '<tool_call>{"name": "read_file", "arguments": {"path": "x"}}</tool_call>'
    _, calls = parse_tool_calls(msg)
    wire = tool_calls_to_wire(calls)
    assert wire == [
        {
            "type": "function",
            "function": {"name": "read_file", "arguments": {"path": "x"}},
        }
    ]


def test_parse_ignores_malformed_json_block() -> None:
    msg = "Final answer: 42.\n<tool_call>{not valid json}</tool_call>"
    text, calls = parse_tool_calls(msg)
    assert "Final answer: 42." in text
    assert calls == []
