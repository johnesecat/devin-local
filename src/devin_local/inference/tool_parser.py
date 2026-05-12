"""Parse tool calls out of free-form assistant text.

Backends that don't speak Ollama's structured `tool_calls` field (AirLLM,
plain HuggingFace) emit tool calls inline as text. Common conventions, in
priority order:

1. Qwen / Hermes / many fine-tunes::

       <tool_call>
       {"name": "...", "arguments": {...}}
       </tool_call>

2. ChatML-style with backticks::

       ```tool_call
       {"name": "...", "arguments": {...}}
       ```

3. Plain top-level JSON whose shape matches a tool call (a last-resort
   fallback used when the model forgets the wrapper but still emits valid
   tool-call JSON).

This module turns any of those into a list of structured `ToolCall` records
and returns the *cleaned* assistant text with the tool-call markup removed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from devin_local.inference.types import ToolCall

# <tool_call>{...}</tool_call> — DOTALL so the JSON can span lines.
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# ```tool_call ... ```  or  ```json ... ``` containing a tool-call shape.
_TOOL_CALL_FENCE = re.compile(
    r"```(?:tool_call|json)\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _looks_like_tool_call(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and obj["name"]
        and ("arguments" in obj or "parameters" in obj)
    )


def _normalize(obj: dict[str, Any]) -> ToolCall:
    name = obj["name"]
    raw_args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {"_raw": raw_args}
    if not isinstance(raw_args, dict):
        raw_args = {"_value": raw_args}
    return ToolCall(name=name, arguments=raw_args, raw=obj)


def _try_load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split assistant text into (clean_content, tool_calls).

    Empty list means the model produced a plain reply with no tool calls.
    The cleaned text has every recognized tool-call block removed; surrounding
    prose is preserved.
    """
    calls: list[ToolCall] = []
    cleaned = text

    # 1) <tool_call>...</tool_call> blocks.
    def _take_tag(match: re.Match[str]) -> str:
        obj = _try_load_json(match.group(1))
        if _looks_like_tool_call(obj):
            calls.append(_normalize(obj))
            return ""
        return match.group(0)

    cleaned = _TOOL_CALL_TAG.sub(_take_tag, cleaned)

    # 2) ```tool_call ... ``` / ```json ... ``` fences.
    def _take_fence(match: re.Match[str]) -> str:
        obj = _try_load_json(match.group(1))
        if _looks_like_tool_call(obj):
            calls.append(_normalize(obj))
            return ""
        return match.group(0)

    cleaned = _TOOL_CALL_FENCE.sub(_take_fence, cleaned)

    # 3) Last-resort: if the *entire* (now-cleaned) message is a JSON object
    #    that looks like a tool call, lift it. Don't do this if we already
    #    parsed something, to avoid double-counting.
    if not calls:
        stripped = cleaned.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            obj = _try_load_json(stripped)
            if _looks_like_tool_call(obj):
                calls.append(_normalize(obj))
                cleaned = ""

    return cleaned.strip(), calls


def tool_calls_to_wire(calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Convert parsed ToolCalls into the same shape Ollama emits, so the
    rest of the agent can treat them identically."""
    return [c.to_wire() for c in calls]
