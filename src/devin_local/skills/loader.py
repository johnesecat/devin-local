"""Load `*.md` skill files with YAML frontmatter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Skill:
    """A skill loaded from a markdown file."""

    name: str
    description: str
    body: str
    scope: list[str] = field(default_factory=list)
    path: Path | None = None

    def to_block(self) -> str:
        return f"### Skill: {self.name}\n{self.description}\n\n{self.body.strip()}"


def parse_skill(path: Path) -> Skill | None:
    """Parse a single markdown file into a Skill. Returns None on failure."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        # No frontmatter — treat the whole file as the body.
        return Skill(
            name=path.stem,
            description=path.stem.replace("-", " ").title(),
            body=raw,
            path=path,
        )
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None
    body = parts[2].strip()
    scope_raw = meta.get("scope") or meta.get("scopes") or ""
    if isinstance(scope_raw, str):
        scope_list = [s.strip().lower() for s in scope_raw.split(",") if s.strip()]
    elif isinstance(scope_raw, list):
        scope_list = [str(s).strip().lower() for s in scope_raw]
    else:
        scope_list = []
    return Skill(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or ""),
        body=body,
        scope=scope_list,
        path=path,
    )


@dataclass
class SkillLoader:
    """Loads and ranks skills from one or more directories."""

    directories: list[Path]
    skills: list[Skill] = field(default_factory=list)

    def reload(self) -> None:
        loaded: list[Skill] = []
        for directory in self.directories:
            if not directory.exists() or not directory.is_dir():
                continue
            for md_file in sorted(directory.rglob("*.md")):
                skill = parse_skill(md_file)
                if skill is not None:
                    loaded.append(skill)
        self.skills = loaded

    def match(self, query: str, limit: int = 3) -> list[Skill]:
        """Return up to `limit` skills whose scope or body matches `query`."""
        if not self.skills:
            return []
        needle = query.lower()
        scored: list[tuple[int, Skill]] = []
        for skill in self.skills:
            score = 0
            for scope_tag in skill.scope:
                if scope_tag and scope_tag in needle:
                    score += 5
            for word in needle.split():
                if word in skill.body.lower():
                    score += 1
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s for _score, s in scored[:limit]]


def load_default_skills(workspace: Path) -> SkillLoader:
    """Convenience: load built-in + workspace skills."""
    builtins = Path(__file__).resolve().parent / "builtins"
    user_skills = workspace / "skills"
    loader = SkillLoader(directories=[builtins, user_skills])
    loader.reload()
    return loader
