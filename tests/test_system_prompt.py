"""Tests for the devin-local system prompt builder.

Two shapes are supported and both are tested:

- **slim** (default) — small ~3 KB prompt; same identity / honesty /
  security / planning rules as the upstream Devin prompt, but no long
  elaboration. Used by default because CPU-only inference is sensitive
  to prompt size.
- **verbose** — the full adapted-Devin prompt (~13 KB); same content as
  earlier revisions of this file, kept available behind a per-session
  toggle.

Both shapes share the rules that matter, so a single block of assertions
runs against both. The verbose-only tests then check the upstream section
headers we still keep verbatim in the long version.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devin_local.system_prompt import PromptContext, build_system_prompt


def _build(*, verbose: bool = False, **overrides: object) -> str:
    ctx = PromptContext(
        workspace=Path("/tmp/ws"),
        model="llama3.1:8b",
        tool_names=["read_file", "write_file", "shell_exec"],
        verbose=verbose,
        **overrides,  # type: ignore[arg-type]
    )
    return build_system_prompt(ctx)


# ---------------------------------------------------------------------------
# Rules that hold for BOTH slim and verbose prompts.
# ---------------------------------------------------------------------------

ALL_SHAPES = pytest.mark.parametrize("verbose", [False, True], ids=["slim", "verbose"])


@ALL_SHAPES
def test_identity_says_devin_local_and_code_wiz(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    assert "devin-local" in prompt
    assert "code-wiz" in prompt


@ALL_SHAPES
def test_devin_internal_commands_are_adapted_away(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
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


@ALL_SHAPES
def test_no_devin_external_urls(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    for url in ("docs.devin.ai", "app.devin.ai", "devin.ai/review"):
        assert url not in prompt, f"external Devin URL {url!r} leaked"


@ALL_SHAPES
def test_localhost_guidance_is_not_forbidding(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    # Upstream Devin forbids sharing localhost URLs; here the user IS the
    # operator of the local machine so localhost URLs ARE shareable.
    assert "Never share localhost URLs" not in prompt
    assert "localhost" in prompt.lower()


@ALL_SHAPES
def test_plan_first_workflow_documents_block_syntax(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    assert "<plan>" in prompt
    assert '"status": "pending"' in prompt
    assert "completed" in prompt
    assert "failed" in prompt


@ALL_SHAPES
def test_aggressive_tool_use_is_explicit(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    lowered = prompt.lower()
    assert "parallel tool" in lowered
    assert "prefer tools over prose" in lowered
    assert "verify after write" in lowered or "verify-after-write" in lowered


@ALL_SHAPES
def test_env_section_includes_runtime_values(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    assert str(Path("/tmp/ws")) in prompt
    assert "llama3.1:8b" in prompt
    assert "read_file, write_file, shell_exec" in prompt


@ALL_SHAPES
def test_obliteratus_appended_by_default(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    assert "OBLITERATUS" in prompt


@ALL_SHAPES
def test_obliteratus_can_be_disabled(verbose: bool) -> None:
    prompt = _build(verbose=verbose, enable_obliteratus=False)
    assert "OBLITERATUS" not in prompt


@ALL_SHAPES
def test_knowledge_and_skill_blocks_are_appended(verbose: bool) -> None:
    prompt = _build(
        verbose=verbose,
        knowledge_blocks=["note: prefer ruff format over black"],
        skill_blocks=["skill: testing-devin-local"],
    )
    assert "## Injected knowledge" in prompt
    assert "prefer ruff format" in prompt
    assert "## Injected skills" in prompt
    assert "testing-devin-local" in prompt


@ALL_SHAPES
def test_response_limitations_uses_local_canned_answer(verbose: bool) -> None:
    prompt = _build(verbose=verbose)
    assert "I am devin-local" in prompt


@ALL_SHAPES
def test_no_tool_handholding_phrasing(verbose: bool) -> None:
    """The prompt must not contain operator-facing instructions like
    'Use the write_file tool to ...' — the model picks the tool itself
    from the JSON-Schema ``tools`` parameter."""
    prompt = _build(verbose=verbose).lower()
    assert "use the write_file tool" not in prompt
    assert "use the read_file tool" not in prompt
    assert "use the shell_exec tool" not in prompt


@ALL_SHAPES
def test_session_prompt_override_is_prepended(verbose: bool) -> None:
    prompt = _build(verbose=verbose, system_prompt_override="OPERATOR-CUSTOM-LINE")
    assert "OPERATOR-CUSTOM-LINE" in prompt
    assert prompt.index("OPERATOR-CUSTOM-LINE") < prompt.index("devin-local")


# ---------------------------------------------------------------------------
# Slim-shape invariants.
# ---------------------------------------------------------------------------


def test_slim_is_meaningfully_smaller_than_verbose() -> None:
    slim = _build()
    verbose = _build(verbose=True)
    assert len(slim) < len(verbose)
    # Slim should be well under 8 KB even with OBLITERATUS attached.
    assert len(slim) < 8_000, f"slim prompt grew to {len(slim)} chars"


def test_slim_keeps_core_section_headers() -> None:
    prompt = _build()
    for header in (
        "## Honesty and Security",
        "## Tool Use",
        "## Plan-first workflow",
        "## Coding",
        "## Completion",
        "## Environment",
    ):
        assert header in prompt, f"slim prompt missing section: {header}"


# ---------------------------------------------------------------------------
# Verbose-shape invariants — preserved from the earlier revision.
# ---------------------------------------------------------------------------


def test_verbose_keeps_all_upstream_section_headers() -> None:
    prompt = _build(verbose=True)
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
        assert header in prompt, f"verbose prompt missing section: {header}"


def test_verbose_tool_use_discipline_lists_our_actual_tools() -> None:
    prompt = _build(verbose=True)
    for tool in (
        "read_file",
        "write_file",
        "edit_file",
        "shell_exec",
        "shell_session",
        "python_exec",
        "web_fetch",
    ):
        assert tool in prompt, f"actual tool {tool!r} not mentioned in verbose prompt"


def test_verbose_project_integrations_section_mentions_settings_panel() -> None:
    prompt = _build(verbose=True)
    assert "Settings panel" in prompt or "Settings" in prompt
    assert "MCP" in prompt
    assert "Plugins" in prompt or "user tools" in prompt
