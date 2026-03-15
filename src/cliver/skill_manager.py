"""
Skill Manager for CLIver.

Discovers, loads, and caches skills from SKILL.md files in directories.
Skills are markdown files with YAML frontmatter that provide domain-specific
instructions and context to the LLM when activated.

Discovery order (project-local overrides user-global):
1. .cliver/skills/<name>/SKILL.md  — project-local
2. {config_dir}/skills/<name>/SKILL.md — user-global
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from cliver.util import get_config_dir

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"


@dataclass
class Skill:
    """A loaded skill parsed from a SKILL.md file."""

    name: str
    description: str
    body: str  # markdown content below the frontmatter
    base_dir: Path  # directory containing SKILL.md
    frontmatter: dict = field(default_factory=dict)



def _parse_skill_md(path: Path) -> Optional[Skill]:
    """Parse a SKILL.md file into a Skill object.

    Expected format:
        ---
        name: skill-name
        description: What it does and when to use it
        ---

        # Markdown body
        ...
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None

    # Split YAML frontmatter from body
    if not content.startswith("---"):
        logger.warning(f"Skill file {path} missing YAML frontmatter (must start with ---)")
        return None

    # Find the closing --- (skip the opening one)
    end_idx = content.index("---", 3)
    if end_idx == -1:
        logger.warning(f"Skill file {path} has unclosed YAML frontmatter")
        return None

    frontmatter_str = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML frontmatter in {path}: {e}")
        return None

    name = frontmatter.get("name")
    description = frontmatter.get("description", "")

    if not name:
        logger.warning(f"Skill file {path} missing required 'name' field in frontmatter")
        return None

    if not description:
        logger.warning(f"Skill '{name}' at {path} has no description — LLM won't know when to activate it")

    return Skill(
        name=name,
        description=description,
        body=body,
        base_dir=path.parent,
        frontmatter=frontmatter,
    )


def _discover_skills_in_dir(skills_dir: Path) -> Dict[str, Skill]:
    """Discover all skills in a directory.

    Each skill is a subdirectory containing a SKILL.md file.
    """
    skills = {}
    if not skills_dir.is_dir():
        return skills

    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_file = child / SKILL_FILENAME
        if not skill_file.is_file():
            continue

        skill = _parse_skill_md(skill_file)
        if skill:
            skills[skill.name] = skill
            logger.debug(f"Discovered skill '{skill.name}' at {skill_file}")

    return skills


class SkillManager:
    """Discovers, loads, and caches skills from SKILL.md files.

    Skills are discovered from two locations in priority order:
    1. Project-local: .cliver/skills/ in the current working directory
    2. User-global: {config_dir}/skills/

    Project-local skills override user-global skills with the same name.
    """

    def __init__(self):
        self._skills: Optional[Dict[str, Skill]] = None

    def _ensure_loaded(self) -> None:
        if self._skills is not None:
            return
        self._skills = {}

        # Load user-global skills first (lower priority)
        global_skills_dir = get_config_dir() / "skills"
        global_skills = _discover_skills_in_dir(global_skills_dir)
        self._skills.update(global_skills)

        # Load project-local skills (higher priority, overrides global)
        local_skills_dir = Path.cwd() / ".cliver" / "skills"
        local_skills = _discover_skills_in_dir(local_skills_dir)
        self._skills.update(local_skills)

        if self._skills:
            logger.info(f"Loaded {len(self._skills)} skill(s): {list(self._skills.keys())}")
        else:
            logger.debug("No skills found")

    def list_skills(self) -> List[Skill]:
        """Return all discovered skills."""
        self._ensure_loaded()
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        self._ensure_loaded()
        return self._skills.get(name)

    def get_skill_names(self) -> List[str]:
        """Get all available skill names."""
        self._ensure_loaded()
        return list(self._skills.keys())

    def format_skill_list(self) -> str:
        """Format available skills as a readable list for LLM display."""
        skills = self.list_skills()
        if not skills:
            return "No skills available."

        lines = ["Available skills:"]
        for skill in skills:
            lines.append(f"  - {skill.name}: {skill.description}")
        return "\n".join(lines)

    def activate_skill(self, name: str) -> str:
        """Activate a skill and return its content for injection into the conversation.

        Returns the skill body prefixed with the base directory path.
        """
        skill = self.get_skill(name)
        if not skill:
            available = self.get_skill_names()
            msg = f"Skill '{name}' not found."
            if available:
                msg += f" Available skills: {', '.join(available)}"
            return msg

        return (
            f"Base directory for this skill: {skill.base_dir}/\n\n"
            f"{skill.body}"
        )

    def reload(self) -> None:
        """Force re-discovery of skills."""
        self._skills = None
        self._ensure_loaded()
