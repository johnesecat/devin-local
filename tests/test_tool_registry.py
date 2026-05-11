"""Tests for the tool registry, base tool helpers, and Ollama schema export."""

from __future__ import annotations

from typing import Any

import pytest

from devin_local.tools.base import Tool, ToolError, ToolResult, error_result, text_result
from devin_local.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "echo back the input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        text = arguments.get("text")
        if not isinstance(text, str):
            raise ToolError("text must be a string")
        return text_result(text)


def test_register_and_dispatch():
    reg = ToolRegistry.empty()
    reg.register(EchoTool())
    res = reg.dispatch("echo", {"text": "hi"})
    assert res.ok
    assert res.output == "hi"


def test_dispatch_unknown_tool_returns_error():
    reg = ToolRegistry.empty()
    res = reg.dispatch("nope", {})
    assert not res.ok
    assert "Unknown tool" in res.output


def test_dispatch_tool_error_becomes_failed_result():
    reg = ToolRegistry.empty()
    reg.register(EchoTool())
    res = reg.dispatch("echo", {"text": 123})  # type: ignore[arg-type]
    assert not res.ok
    assert "must be a string" in res.output


def test_dispatch_unhandled_exception_is_caught():
    class Boom(Tool):
        name = "boom"
        description = "raises"
        parameters = {"type": "object", "properties": {}}

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            raise RuntimeError("kaboom")

    reg = ToolRegistry.empty()
    reg.register(Boom())
    res = reg.dispatch("boom", {})
    assert not res.ok
    assert "kaboom" in res.output


def test_duplicate_registration_raises():
    reg = ToolRegistry.empty()
    reg.register(EchoTool())
    with pytest.raises(ToolError):
        reg.register(EchoTool())


def test_to_ollama_schemas_shape():
    reg = ToolRegistry.empty()
    reg.register(EchoTool())
    schemas = reg.to_ollama_schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert "parameters" in schema["function"]


def test_error_result_helper_marks_not_ok():
    res = error_result("bad")
    assert res.ok is False
    assert res.output == "bad"
