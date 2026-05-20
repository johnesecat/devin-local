"""Agent planning: structured task plans that mirror Devin's todo workflow.

The agent emits a short, structured plan at the start of each new user turn
(and after major task pivots) by wrapping a JSON list in a sentinel block::

    <plan>
    [{"text": "Read pyproject.toml", "status": "pending"}, ...]
    </plan>

The wrapper is hidden from the human-visible bubble (the tool-card pane
filters it out) and forwarded to :class:`PlanPane` in the GUI. Each step
transitions ``pending -> in_progress -> completed`` (or ``failed``) as the
agent works.

This module owns:

- :class:`PlanStep` — one todo entry
- :func:`parse_plan_block` — extract a plan from raw assistant text
- :func:`strip_plan_block` — return assistant text with the <plan> tag removed
- :func:`render_plan_for_prompt` — render a plan in human-friendly form for
  re-injection into the prompt next turn (so the model sees its own todos)

It is intentionally side-effect free; the agent loop owns mutation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_PLAN_RE = re.compile(r"(?ms)<plan>\s*(\[.*?\])\s*</plan>")
_VALID_STATUS = {"pending", "in_progress", "completed", "failed"}


@dataclass
class PlanStep:
    """One step in an agent plan."""

    text: str
    status: str = "pending"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "status": self.status, "note": self.note}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> PlanStep:
        text = str(obj.get("text") or obj.get("step") or obj.get("title") or "").strip()
        status = str(obj.get("status") or "pending").strip() or "pending"
        if status not in _VALID_STATUS:
            status = "pending"
        return cls(text=text, status=status, note=str(obj.get("note") or "").strip())


@dataclass
class Plan:
    """An ordered list of plan steps."""

    steps: list[PlanStep] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.steps

    def progress(self) -> tuple[int, int]:
        done = sum(1 for s in self.steps if s.status == "completed")
        return done, len(self.steps)

    def merge(self, other: Plan) -> None:
        """Merge `other` into self by text match; new steps appended at end.

        Used when the model emits a refined plan mid-task. Existing step
        statuses are preserved unless `other` explicitly transitions them.
        """
        by_text = {s.text: s for s in self.steps}
        for new in other.steps:
            if new.text in by_text:
                existing = by_text[new.text]
                # Only allow forward transitions; never demote completed -> pending.
                if _rank(new.status) >= _rank(existing.status):
                    existing.status = new.status
                if new.note:
                    existing.note = new.note
            else:
                self.steps.append(new)
        # If `other` is shorter than self, drop trailing self entries that
        # are still pending — the model has decided they aren't needed.
        if len(other.steps) < len(self.steps):
            kept: list[PlanStep] = []
            other_texts = {s.text for s in other.steps}
            for step in self.steps:
                if step.status in ("completed", "failed") or step.text in other_texts:
                    kept.append(step)
            self.steps = kept

    def mark_in_progress(self, text: str) -> None:
        for step in self.steps:
            if step.text == text:
                step.status = "in_progress"
                return

    def mark_completed(self, text: str) -> None:
        for step in self.steps:
            if step.text == text:
                step.status = "completed"
                return


def _rank(status: str) -> int:
    return {"pending": 0, "in_progress": 1, "completed": 2, "failed": 2}.get(status, 0)


def parse_plan_block(text: str) -> Plan | None:
    """Find a ``<plan>[...]</plan>`` block in `text` and parse it.

    Returns None if no block is found or parsing fails. Tolerant of trailing
    commas (commonly emitted by smaller models) by stripping them before
    json.loads.
    """
    match = _PLAN_RE.search(text)
    if match is None:
        return None
    raw = match.group(1).strip()
    raw = re.sub(r",(\s*[\]\}])", r"\1", raw)  # strip trailing commas
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    steps: list[PlanStep] = []
    for entry in data:
        if isinstance(entry, str):
            steps.append(PlanStep(text=entry.strip()))
        elif isinstance(entry, dict):
            step = PlanStep.from_dict(entry)
            if step.text:
                steps.append(step)
    if not steps:
        return None
    return Plan(steps=steps)


def strip_plan_block(text: str) -> str:
    """Return `text` with any ``<plan>...</plan>`` block removed."""
    return _PLAN_RE.sub("", text).strip()


def render_plan_for_prompt(plan: Plan) -> str:
    """Format `plan` as a human-readable todo list for re-injection."""
    if plan.is_empty():
        return ""
    lines = ["Current plan:"]
    for i, step in enumerate(plan.steps, start=1):
        glyph = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "completed": "[x]",
            "failed": "[!]",
        }.get(step.status, "[ ]")
        note = f"  ({step.note})" if step.note else ""
        lines.append(f"  {i}. {glyph} {step.text}{note}")
    return "\n".join(lines)
