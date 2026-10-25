"""CLIver's eight skill discovery sources.

cliver-core deliberately owns no discovery: ``SkillManager`` reads only the
directories an application supplies.  This module supplies CLIver's
conventions — the builtin skills shipped with the package plus the global
and project-local directories shared with other coding agents — in
ascending priority order.  Later sources override earlier ones with the
same skill name, as in the legacy CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cliver_core.skill import Skill, SkillManager

from cliver.paths import get_config_dir

BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtin"

__all__ = [
    "BUILTIN_SKILLS_DIR",
    "SkillSource",
    "create_skill_manager",
    "default_skill_sources",
    "source_for_skill",
]


@dataclass(frozen=True)
class SkillSource:
    """One discovery directory and the label shown in CLI listings."""

    path: Path
    label: str


def default_skill_sources(
    *,
    config_dir: Path | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> list[SkillSource]:
    """Return the eight CLIver skill sources, lowest priority first.

    *config_dir*, *cwd*, and *home* are injection points for tests; in the
    application they default to :func:`cliver.paths.get_config_dir`, the
    process working directory, and the user's home directory.
    """
    home_dir = Path.home() if home is None else Path(home).expanduser()
    config_path = get_config_dir(home=home_dir) if config_dir is None else Path(config_dir).expanduser()
    project_dir = Path.cwd() if cwd is None else Path(cwd).expanduser()

    return [
        SkillSource(BUILTIN_SKILLS_DIR, "builtin"),
        SkillSource(config_path / "skills", "global (cliver)"),
        SkillSource(home_dir / ".agents" / "skills", "global (agents)"),
        SkillSource(project_dir / ".cliver" / "skills", "project (.cliver)"),
        SkillSource(project_dir / ".agent" / "skills", "project (.agent)"),
        SkillSource(project_dir / ".claude" / "skills", "project (.claude compat)"),
        SkillSource(project_dir / ".gemini" / "skills", "project (.gemini compat)"),
        SkillSource(project_dir / ".qwen" / "skills", "project (.qwen compat)"),
    ]


def create_skill_manager(
    *,
    config_dir: Path | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> SkillManager:
    """Build a core ``SkillManager`` over the eight CLIver sources."""
    return SkillManager(
        sources=[source.path for source in default_skill_sources(config_dir=config_dir, cwd=cwd, home=home)]
    )


def source_for_skill(skill: Skill, sources: list[SkillSource]) -> SkillSource | None:
    """Return the source a loaded skill came from, or ``None`` if unknown.

    A source is either a container of skill directories or a single skill
    directory, so the skill's base directory matches the source itself or
    its immediate child.  Paths are compared after resolution so symlinked
    source directories still match.
    """
    base_dir = skill.base_dir.resolve()
    for source in sources:
        source_dir = source.path.expanduser().resolve()
        if base_dir == source_dir or base_dir.parent == source_dir:
            return source
    return None
