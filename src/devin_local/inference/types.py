"""Shared types for inference backends.

`ChatMessage` is the canonical message shape — every backend produces and
consumes these regardless of the underlying transport (Ollama JSON, HF tokens,
AirLLM strings). The Ollama-specific module re-exports `ChatMessage` from here
for backwards compatibility with code that imported it before the
`inference/` package existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a conversation.

    `role` is one of "system", "user", "assistant", "tool".
    `tool_calls` is populated only on assistant messages that requested tools.
    `name` is set on tool-response messages to identify which tool ran.
    """

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class ToolCall:
    """A parsed tool invocation requested by the model.

    Some backends speak the OpenAI/Ollama wire shape natively
    (`{"function": {"name": ..., "arguments": ...}}`); others emit text we
    have to parse. Either way the agent only ever sees `ToolCall` objects.
    """

    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ToolDefinition:
    """A tool the backend is allowed to call. JSON-schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
