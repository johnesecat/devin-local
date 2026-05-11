"""Skill loader.

A skill is a markdown file with YAML frontmatter, describing reference
knowledge or a procedure the agent should follow when its scope matches the
current task. Example:

    ---
    name: run-pytest
    description: How to run the project's test suite.
    scope: testing, pytest
    ---

    From the repository root:

        python -m pytest -q

Skills are loaded from:
  - `src/devin_local/skills/builtins/` (shipped with the package)
  - `skills/` at the workspace root (user-supplied)
"""

from __future__ import annotations

from devin_local.skills.loader import Skill, SkillLoader, load_default_skills

__all__ = ["Skill", "SkillLoader", "load_default_skills"]
