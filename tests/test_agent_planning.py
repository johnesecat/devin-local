"""Tests for the agent_planning module."""

from __future__ import annotations

from devin_local.agent_planning import (
    Plan,
    PlanStep,
    parse_plan_block,
    render_plan_for_prompt,
    strip_plan_block,
)


def test_parse_plan_block_extracts_list_of_dicts() -> None:
    text = """
    Sure, here's my approach.

    <plan>
    [
      {"text": "Read pyproject.toml", "status": "pending"},
      {"text": "Write hello.py", "status": "in_progress"}
    ]
    </plan>

    Now I'll start.
    """
    plan = parse_plan_block(text)
    assert plan is not None
    assert len(plan.steps) == 2
    assert plan.steps[0].text == "Read pyproject.toml"
    assert plan.steps[0].status == "pending"
    assert plan.steps[1].status == "in_progress"


def test_parse_plan_block_accepts_string_entries() -> None:
    text = '<plan>["step one", "step two"]</plan>'
    plan = parse_plan_block(text)
    assert plan is not None
    assert [s.text for s in plan.steps] == ["step one", "step two"]
    assert all(s.status == "pending" for s in plan.steps)


def test_parse_plan_block_returns_none_when_missing() -> None:
    assert parse_plan_block("no plan here") is None
    assert parse_plan_block("<plan>not-json</plan>") is None
    assert parse_plan_block("<plan>{}</plan>") is None


def test_parse_plan_block_tolerates_trailing_commas() -> None:
    text = '<plan>[{"text": "do x", "status": "pending",},]</plan>'
    plan = parse_plan_block(text)
    assert plan is not None
    assert len(plan.steps) == 1


def test_strip_plan_block_removes_block() -> None:
    text = 'before\n<plan>[{"text": "x", "status": "pending"}]</plan>\nafter'
    out = strip_plan_block(text)
    assert "<plan>" not in out
    assert "before" in out
    assert "after" in out


def test_plan_merge_preserves_completed_status() -> None:
    base = Plan(
        steps=[PlanStep(text="A", status="completed"), PlanStep(text="B", status="pending")]
    )
    update = Plan(
        steps=[PlanStep(text="A", status="pending"), PlanStep(text="B", status="in_progress")]
    )
    base.merge(update)
    # A should stay completed (never demote)
    assert base.steps[0].status == "completed"
    # B should advance to in_progress
    assert base.steps[1].status == "in_progress"


def test_plan_merge_appends_new_steps() -> None:
    base = Plan(steps=[PlanStep(text="A", status="completed")])
    update = Plan(
        steps=[PlanStep(text="A", status="completed"), PlanStep(text="C", status="pending")]
    )
    base.merge(update)
    assert [s.text for s in base.steps] == ["A", "C"]


def test_render_plan_for_prompt() -> None:
    plan = Plan(
        steps=[
            PlanStep(text="A", status="completed"),
            PlanStep(text="B", status="in_progress"),
            PlanStep(text="C", status="pending"),
        ]
    )
    rendered = render_plan_for_prompt(plan)
    assert "[x] A" in rendered
    assert "[~] B" in rendered
    assert "[ ] C" in rendered
