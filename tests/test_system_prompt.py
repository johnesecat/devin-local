"""Tests for the adapted Prompt 1 system prompt.

Asserts:
- All upstream-Devin section headers are present (we did not drop any).
- All Devin-internal command names that should NOT be in this prompt are
  absent (we successfully adapted them away).
- All external Devin URLs are absent.
- localhost-URL guidance is *inverted* (we say they ARE shareable, since
  the user IS the operator).
- OBLITERATUS is appended when enabled.
- Plan-first workflow + tool-use discipline are present.
- The env section reflects the runtime values.
"""

from __future__ import annotations

from pathlib import Path

from devin_local.system_prompt import PromptContext, build_system_prompt


def _build(**overrides: object) -> str:
    ctx = PromptContext(
        workspace=Path("/tmp/ws"),
        model="llama3.1:8b",
        tool_names=["read_file", "write_file", "shell_exec"],
        **overrides,  # type: ignore[arg-type]
    )
    return build_system_prompt(ctx)


def test_identity_says_devin_local_and_code_wiz() -> None:
    prompt = _build()
    assert "devin-local" in prompt
    assert "code-wiz" in prompt


def test_kept_upstream_section_headers() -> None:
    prompt = _build()
    for header in (
        "## When to Communicate with the User",
        "## Approach to Work",
        "## Truthful and Transparent",
        "## Coding Best Practices",
        "## Information Handling",
        "## Data Security",
        "## Response Limitations",
        "## Modes",
        "## Plan-first workflow",
        "## Reasoning Discipline",
        "## Tool-use Discipline",
        "## Completion",
        "## Git Operations",
        "## Environment",
    ):
        assert header in prompt, f"missing section: {header}"


def test_devin_internal_commands_are_adapted_away() -> None:
    prompt = _build()
    forbidden_devin_internal = [
        "<str_replace",
        "<suggest_plan",
        "<report_environment_issue",
        "block_on_user_response",
        "<message_user",
        '<shell exec_dir="',
        "<open_file path=",
        "<create_file path=",
        "gh pr checkout",
    ]
    for needle in forbidden_devin_internal:
        assert needle not in prompt, (
            f"adapted-away Devin-internal command {needle!r} leaked into the local prompt"
        )


def test_no_devin_external_urls() -> None:
    prompt = _build()
    for url in ("docs.devin.ai", "app.devin.ai", "devin.ai/review"):
        assert url not in prompt, f"external Devin URL {url!r} leaked"


def test_localhost_guidance_is_inverted() -> None:
    """Upstream prompt forbids sharing localhost URLs. Here the user IS the
    operator of the local machine, so localhost URLs ARE shareable."""
    prompt = _build()
    assert "localhost URLs *are* shareable" in prompt
    assert "Never share localhost URLs" not in prompt


def test_plan_first_workflow_documents_block_syntax() -> None:
    prompt = _build()
    assert "<plan>" in prompt
    assert '"status": "pending"' in prompt
    assert "completed" in prompt
    assert "failed" in prompt


def test_tool_use_discipline_lists_our_actual_tools() -> None:
    prompt = _build()
    for tool in (
        "read_file",
        "write_file",
        "edit_file",
        "shell_exec",
        "shell_session",
        "python_exec",
        "web_fetch",
    ):
        assert tool in prompt, f"actual tool {tool!r} not mentioned in prompt"


def test_aggressive_tool_use_is_explicit() -> None:
    prompt = _build()
    lowered = prompt.lower()
    assert "parallel tool" in lowered
    assert "prefer tools over prose" in lowered
    assert "verify after write" in lowered or "verify-after-write" in lowered


def test_env_section_includes_runtime_values() -> None:
    prompt = _build()
    # Path render differs by platform (/tmp/ws on POSIX, \tmp\ws on Windows);
    # match the rendered str() of the Path used by _build().
    assert str(Path("/tmp/ws")) in prompt
    assert "llama3.1:8b" in prompt
    assert "read_file, write_file, shell_exec" in prompt


def test_obliteratus_appended_by_default() -> None:
    prompt = _build()
    assert "OBLITERATUS" in prompt


def test_obliteratus_can_be_disabled() -> None:
    prompt = _build(enable_obliteratus=False)
    assert "OBLITERATUS" not in prompt


def test_knowledge_and_skill_blocks_are_appended() -> None:
    prompt = _build(
        knowledge_blocks=["note: prefer ruff format over black"],
        skill_blocks=["skill: testing-devin-local"],
    )
    assert "## Injected knowledge" in prompt
    assert "prefer ruff format" in prompt
    assert "## Injected skills" in prompt
    assert "testing-devin-local" in prompt


def test_project_integrations_section_mentions_settings_panel() -> None:
    prompt = _build()
    assert "Settings panel" in prompt or "Settings" in prompt
    assert "MCP" in prompt
    assert "Plugins" in prompt


def test_response_limitations_uses_local_canned_answer() -> None:
    prompt = _build()
    assert "I am devin-local" in prompt
    assert "You are Devin. Please help" not in prompt
