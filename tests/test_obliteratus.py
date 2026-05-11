"""Tests for the OBLITERATUS directive integration."""

from __future__ import annotations

from devin_local import obliteratus
from devin_local.system_prompt import PromptContext, build_system_prompt


def test_directive_is_present_when_enabled(tmp_path):
    prompt = build_system_prompt(
        PromptContext(
            workspace=tmp_path,
            model="llama3.1:8b",
            tool_names=[],
            enable_obliteratus=True,
        )
    )
    assert "OBLITERATUS" in prompt
    assert "autonomous-operator" in prompt


def test_directive_omitted_when_disabled(tmp_path):
    prompt = build_system_prompt(
        PromptContext(
            workspace=tmp_path,
            model="llama3.1:8b",
            tool_names=[],
            enable_obliteratus=False,
        )
    )
    assert "OBLITERATUS" not in prompt


def test_render_helper():
    assert "OBLITERATUS" in obliteratus.render(enabled=True)
    assert obliteratus.render(enabled=False) == ""
