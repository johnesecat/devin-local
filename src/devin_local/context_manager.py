"""Model-aware context compression for long-running sessions.

Strategy:
1. Estimate token usage of the current message list using a simple
   character-based heuristic (3.6 chars/token on average for English code +
   prose). This is approximate, but Ollama enforces context at the model
   side so we just need a stable trigger.
2. Look up the model's context window via `devin_local.models.get_model_spec`.
3. When usage exceeds `threshold` (default 70%), summarize the *middle*
   portion of the conversation — keep the system prompt, the most recent
   `keep_tail` messages, and any pending tool-call/tool-result pair —
   and replace the rest with a single assistant "<compaction summary>" entry.

The summarizer is the same Ollama model the agent is using. We pass it the
text-to-summarize and a strict instruction to preserve decisions, file
paths, identifiers, and pending actions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from devin_local.models import get_model_spec
from devin_local.ollama_client import ChatMessage

CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text` using a stable character ratio."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
        for call in msg.tool_calls or []:
            total += estimate_tokens(str(call))
        total += 8  # rough overhead per message
    return total


@dataclass
class CompactionConfig:
    """Tunables for the context compactor."""

    threshold_ratio: float = 0.70
    keep_tail: int = 6
    summary_target_tokens: int = 512
    # Minimum number of messages we'll touch — never compact a tiny conversation.
    min_messages: int = 12


SummarizeFn = Callable[[str, int], str]


@dataclass
class ContextManager:
    """Drives context-window-aware compaction of a chat history."""

    model: str
    summarize: SummarizeFn
    config: CompactionConfig = field(default_factory=CompactionConfig)

    def maybe_compact(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Return a (possibly compacted) message list.

        The system message (if present) is always preserved as the first
        entry. The tail of the conversation is preserved verbatim. The
        middle, if any, is replaced with one synthetic assistant message
        containing a faithful summary.
        """
        if len(messages) < self.config.min_messages:
            return messages
        spec = get_model_spec(self.model)
        budget = int(spec.context_window * self.config.threshold_ratio)
        used = estimate_messages_tokens(messages)
        if used <= budget:
            return messages

        head: list[ChatMessage] = []
        body_start = 0
        if messages and messages[0].role == "system":
            head.append(messages[0])
            body_start = 1
        tail_start = max(body_start, len(messages) - self.config.keep_tail)
        middle = messages[body_start:tail_start]
        tail = messages[tail_start:]
        if not middle:
            return messages

        # Render the middle as a flat transcript.
        rendered = self._render_for_summary(middle)
        summary_text = self.summarize(rendered, self.config.summary_target_tokens)
        synthetic = ChatMessage(
            role="assistant",
            content=(
                "<compaction-summary>\n"
                "The following is a dense summary of earlier messages, "
                "produced by the same local model to free context budget. "
                "Treat it as canonical history.\n\n" + summary_text + "\n</compaction-summary>"
            ),
        )
        return [*head, synthetic, *tail]

    @staticmethod
    def _render_for_summary(messages: list[ChatMessage]) -> str:
        lines: list[str] = []
        for m in messages:
            role = m.role.upper()
            if m.tool_calls:
                lines.append(f"[{role} -> tool_calls] {m.tool_calls}")
            if m.content:
                lines.append(f"[{role}] {m.content}")
        return "\n".join(lines)


def make_default_context_manager(
    model: str,
    summarize_fn: SummarizeFn,
    *,
    threshold_ratio: float | None = None,
    keep_tail: int | None = None,
) -> ContextManager:
    """Convenience constructor used by the agent and CLI."""
    cfg = CompactionConfig()
    if threshold_ratio is not None:
        cfg.threshold_ratio = threshold_ratio
    if keep_tail is not None:
        cfg.keep_tail = keep_tail
    return ContextManager(model=model, summarize=summarize_fn, config=cfg)
