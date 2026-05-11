"""Base abstractions for tools the agent can call.

A tool is a single capability the agent can invoke. Each tool advertises a
JSON-Schema for its arguments (compatible with Ollama's `tools` array) and a
`run` method that returns a `ToolResult`. Tools must never raise into the
agent loop — failures are returned as `ToolResult(ok=False, ...)`.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """Raised internally by a tool to short-circuit execution.

    The `ToolRegistry` catches this and converts it into a failed
    `ToolResult` so the agent never sees a raised exception.
    """


@dataclass
class ToolResult:
    """Outcome of a single tool invocation.

    `output` is the text the agent will see. `data` is optional structured
    metadata callers may consume (tests, plugins) but is not surfaced into
    the model context.
    """

    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_chat_payload(self) -> str:
        """Serialize the result for inclusion as a tool message."""
        if self.ok:
            return self.output
        return f"ERROR: {self.output}"


class Tool(abc.ABC):
    """Abstract base for every tool. Subclasses must set name/description/schema."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    dangerous: bool = False  # Set True for tools requiring confirmation.

    def to_ollama_schema(self) -> dict[str, Any]:
        """Render the tool in the format Ollama's chat API expects.

        Ollama follows OpenAI's tool-calling shape: an object with a `function`
        field carrying `name`, `description`, and a JSON-Schema `parameters`.
        """
        if not self.name:
            raise ToolError("Tool subclass must define a non-empty name")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }

    @abc.abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with validated `arguments` and return a result."""


def text_result(message: str, **data: Any) -> ToolResult:
    """Helper: build a successful text result."""
    return ToolResult(ok=True, output=message, data=dict(data))


def error_result(message: str, **data: Any) -> ToolResult:
    """Helper: build a failed result with a human-readable error."""
    return ToolResult(ok=False, output=message, data=dict(data))


def truncate(text: str, max_chars: int = 12_000) -> str:
    """Truncate `text` to `max_chars` with a clear marker so the agent knows."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [truncated {omitted} characters] ...\n{tail}"


def require_str(args: dict[str, Any], key: str) -> str:
    """Pull a required string parameter, raising `ToolError` if absent."""
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"Missing required string argument: {key!r}")
    return value


def optional_str(args: dict[str, Any], key: str, default: str = "") -> str:
    value = args.get(key, default)
    if value is None:
        return default
    return str(value)


def json_dump(obj: Any) -> str:
    """Stable JSON dump for tool outputs."""
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
