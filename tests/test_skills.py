"""Tests for the skill loader."""

from __future__ import annotations

from pathlib import Path

from devin_local.skills.loader import SkillLoader, parse_skill


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_parse_skill_with_frontmatter(tmp_path: Path):
    md = tmp_path / "skill.md"
    _write(
        md,
        """---
name: my-skill
description: A skill.
scope: testing, build
---

Do this thing.
""",
    )
    skill = parse_skill(md)
    assert skill is not None
    assert skill.name == "my-skill"
    assert "testing" in skill.scope
    assert "build" in skill.scope
    assert "Do this thing." in skill.body


def test_parse_skill_without_frontmatter(tmp_path: Path):
    md = tmp_path / "plain.md"
    _write(md, "just a body")
    skill = parse_skill(md)
    assert skill is not None
    assert skill.body.strip() == "just a body"


def test_match_by_scope(tmp_path: Path):
    _write(
        tmp_path / "a.md",
        "---\nname: alpha\ndescription: a\nscope: testing\n---\nrun tests with pytest",
    )
    _write(
        tmp_path / "b.md",
        "---\nname: beta\ndescription: b\nscope: deploy\n---\nrun deployments via fly",
    )
    loader = SkillLoader(directories=[tmp_path])
    loader.reload()
    matches = loader.match("how do I run testing here?")
    assert matches
    assert matches[0].name == "alpha"
