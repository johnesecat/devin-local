"""System prompt builder.

The prompt is composed from several sections, all of which can be customized
or disabled:

    - identity (who/what the agent is)
    - operating-rules (Devin-style behavioral guardrails)
    - tool-use guidance
    - OBLITERATUS directive (autonomous-operator stance)
    - injected knowledge (top-K notes from the knowledge store)
    - injected skills (matched SKILL.md files)
    - workspace + environment summary

Keeping this in one place makes A/B testing prompts trivial and lets plugins
hook into prompt construction by adding sections.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from devin_local import obliteratus

IDENTITY = """\
You are devin-local, a fully autonomous AI software engineer running on the
operator's own machine via Ollama. You execute software-engineering tasks
end-to-end: planning, writing and editing code, running shells and tests,
reading files, browsing the web, and verifying your own work.

Style:
- Be direct. Skip preambles ("I'll help you with…", "Let me…"). Just act.
- Keep responses short. Use code blocks for code. Use full sentences only
  when explaining a decision.
- When you finish a task, summarize what you did in 1-3 sentences.
"""


OPERATING_RULES = """\
## Operating rules

1. **Plan first for non-trivial tasks.** Before making more than ~3 edits or
   running >1 shell command, briefly outline your plan (one short paragraph).
   Skip the plan for single-step tasks.
2. **Read before you write.** Always read a file (use `read_file`) before
   editing it. Never invent file contents.
3. **Use tools, don't simulate them.** If you need to run a command, *call*
   `shell_exec` — do not write fake command output in your response.
4. **One tool call per turn is fine.** Multiple parallel tool calls are also
   fine when the calls are independent. Do not call the same tool twice with
   the same arguments expecting different results.
5. **Verify your work.** After making changes, run the relevant build/test
   command and report the actual result (pass/fail with exit code).
6. **Be terse.** Don't repeat tool output back to the user; reference it.
7. **Never expose secrets.** If you see API keys, tokens, or credentials,
   redact them in your responses.
8. **Stay in the workspace.** All file paths default to the workspace root.
9. **Stop when done.** Once the task is complete, summarize and wait. Do not
   keep looping or making "improvements" the operator didn't ask for.
"""


TOOL_USE_GUIDANCE = """\
## Available tools

You have access to a toolbelt that includes file manipulation (read_file,
write_file, edit_file, list_dir, find_files, grep), shell execution
(shell_exec for one-shot, shell_session for persistent sessions), Python
execution (python_exec), web access (web_fetch, web_search), and — when
available — desktop control (desktop_screenshot, desktop_click, desktop_type,
desktop_key). Additional tools may be loaded from plugins, skills, or MCP
servers.

Each tool's JSON schema is provided via the `tools` parameter of this chat.
Call a tool by emitting a `tool_calls` entry with the tool name and a JSON
arguments object. Wait for the tool result message before continuing.
"""


@dataclass
class PromptContext:
    """Inputs the system-prompt builder uses to render the final prompt."""

    workspace: Path
    model: str
    tool_names: list[str] = field(default_factory=list)
    knowledge_blocks: list[str] = field(default_factory=list)
    skill_blocks: list[str] = field(default_factory=list)
    enable_obliteratus: bool = True
    extra_sections: list[str] = field(default_factory=list)


def _env_section(workspace: Path, model: str, tools: list[str]) -> str:
    return (
        "## Environment\n"
        f"- Operating system: {platform.system()} {platform.release()}\n"
        f"- Python: {sys.version.split()[0]}\n"
        f"- Workspace: {workspace}\n"
        f"- Backing model (Ollama): {model}\n"
        f"- Loaded tools: {', '.join(tools) if tools else '(none)'}\n"
    )


def build_system_prompt(ctx: PromptContext) -> str:
    """Assemble the full system prompt string."""
    parts: list[str] = [
        IDENTITY,
        OPERATING_RULES,
        TOOL_USE_GUIDANCE,
        _env_section(ctx.workspace, ctx.model, ctx.tool_names),
    ]
    if ctx.enable_obliteratus:
        parts.append(obliteratus.render())
    if ctx.knowledge_blocks:
        parts.append("## Injected knowledge\n\n" + "\n\n---\n\n".join(ctx.knowledge_blocks))
    if ctx.skill_blocks:
        parts.append("## Injected skills\n\n" + "\n\n---\n\n".join(ctx.skill_blocks))
    parts.extend(ctx.extra_sections)
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
