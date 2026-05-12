"""Render a tool-calling chat template for backends that don't natively support tools.

For Ollama, the daemon owns the chat template — we just send a structured
messages array and `tools` array. For AirLLM and HuggingFace transformers, we
have to apply the chat template ourselves and put the tool catalog into the
system prompt.

We use the Qwen / Hermes / many-fine-tunes convention because (a) it's the
template most current open-weights instruct models actually understand, and
(b) it's the most common format already targeted by `tool_parser.py`::

    # Tools

    You may call one or more functions. Function signatures within <tools>:
    <tools>
    {"type":"function","function":{"name":"write_file", ...}}
    {"type":"function","function":{"name":"shell_exec", ...}}
    </tools>

    For each function call, return JSON inside <tool_call></tool_call>:
    <tool_call>{"name":"write_file","arguments":{"path":"hello.py", ...}}</tool_call>

The output of `format_prompt()` is plain text ready to feed to a tokenizer.
"""

from __future__ import annotations

import json
from typing import Any

from devin_local.inference.types import ChatMessage

_TOOL_PREAMBLE = """\

# Tools

You may call one or more functions to assist with the user query. You are
provided with function signatures within <tools></tools>:
<tools>
{tools_block}
</tools>

For each function call, return a JSON object with the function name and
arguments within <tool_call></tool_call> tags with NO other text. Do not
include backticks or ```json. Use exactly this shape:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call>

When you are done calling tools and ready to respond to the user, write a
plain final message with NO <tool_call> tags."""


def _render_tools_block(tools: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)


def render_system_with_tools(system_content: str, tools: list[dict[str, Any]] | None) -> str:
    """Append a tool catalog to the system message, if tools are provided."""
    if not tools:
        return system_content
    return (
        (system_content or "").rstrip()
        + "\n"
        + _TOOL_PREAMBLE.format(
            tools_block=_render_tools_block(tools),
        )
    )


def _role_tag(role: str) -> str:
    # Map our roles to ChatML <|im_start|>...<|im_end|>. Tool-result messages
    # come back as `tool` role; we render them as a user-side observation so
    # base models without a `tool` role still understand them.
    return {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "user",
    }.get(role, "user")


def format_chatml_prompt(
    messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None
) -> str:
    """Render the conversation in ChatML and return a single prompt string.

    The first `system` message (if any) is augmented with the tool catalog.
    Tool-result messages are rendered as user observations like::

        <observation tool="write_file">
        {"ok": true, ...}
        </observation>
    """
    out: list[str] = []
    saw_system = False
    for msg in messages:
        if msg.role == "system":
            content = render_system_with_tools(msg.content, tools if not saw_system else None)
            saw_system = True
            out.append(f"<|im_start|>system\n{content}<|im_end|>")
            continue
        if msg.role == "tool":
            tool_name = msg.name or "tool"
            payload = msg.content
            out.append(
                f'<|im_start|>user\n<observation tool="{tool_name}">\n{payload}\n'
                f"</observation><|im_end|>"
            )
            continue
        if msg.role == "assistant" and msg.tool_calls:
            # Echo the assistant's prior tool-call request verbatim so the model
            # sees its own action in the transcript.
            parts: list[str] = []
            if msg.content:
                parts.append(msg.content)
            for call in msg.tool_calls:
                fn = call.get("function") if isinstance(call, dict) else {}
                if isinstance(fn, dict):
                    parts.append(
                        "<tool_call>"
                        + json.dumps(
                            {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})},
                            ensure_ascii=False,
                        )
                        + "</tool_call>"
                    )
            out.append(f"<|im_start|>assistant\n{''.join(parts)}<|im_end|>")
            continue
        role = _role_tag(msg.role)
        out.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")
    if not saw_system and tools:
        # No system message provided but tools requested — inject one.
        sys_text = render_system_with_tools("", tools)
        out.insert(0, f"<|im_start|>system\n{sys_text}<|im_end|>")
    out.append("<|im_start|>assistant\n")
    return "\n".join(out)
