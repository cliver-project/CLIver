"""
Skill Manager for CLIver.

Discovers, loads, and caches skills from SKILL.md files in directories,
following the Agent Skills specification (https://agentskills.io/specification).

Discovery order (later sources override earlier ones with the same name):
0. {package}/cliver/skills/       — builtin (shipped with CLIver)
1. {config_dir}/skills/           — user-global (CLIver)
2. ~/.agents/skills/              — user-global (agent-agnostic)
3. .cliver/skills/                — project-local (CLIver)
4. .agent/skills/                 — project-local (agent-agnostic)
5. .claude/skills/                — project-local (Claude Code compat)
6. .gemini/skills/                — project-local (Gemini compat)
7. .qwen/skills/                  — project-local (Qwen Code compat)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from cliver.util import get_config_dir

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"

# ---------------------------------------------------------------------------
# Name validation regex per spec:
#   1-64 chars, lowercase a-z, digits 0-9, single hyphens, no leading/trailing hyphen
# ---------------------------------------------------------------------------
_VALID_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){0,63}$")


@dataclass
class SkillValidationResult:
    """Result of validating a skill against the Agent Skills spec."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass
class Skill:
    """A loaded skill parsed from a SKILL.md file."""

    name: str
    description: str
    body: str  # markdown content below the frontmatter
    base_dir: Path  # directory containing SKILL.md

    # Full frontmatter dict (for forward-compat with unknown fields)
    frontmatter: dict = field(default_factory=dict)

    # Optional spec fields
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    allowed_tools: Optional[List[str]] = None

    # Discovery source label (e.g. "project (.cliver)", "global", "claude-compat")
    source: str = ""


def validate_skill_name(name: str) -> List[str]:
    """Validate a skill name against the Agent Skills spec.

    Returns a list of error messages (empty if valid).
    """
    errors = []
    if not name:
        errors.append("name is empty")
        return errors
    if len(name) > 64:
        errors.append(f"name exceeds 64 characters ({len(name)})")
    if not _VALID_NAME_RE.match(name):
        reasons = []
        if name != name.lower():
            reasons.append("must be lowercase")
        if name.startswith("-") or name.endswith("-"):
            reasons.append("must not start or end with hyphen")
        if "--" in name:
            reasons.append("must not contain consecutive hyphens")
        if re.search(r"[^a-z0-9\-]", name):
            reasons.append("only lowercase letters, digits, and hyphens allowed")
        errors.append(f"invalid name '{name}': {', '.join(reasons) if reasons else 'does not match spec pattern'}")
    return errors


def validate_skill(skill: "Skill") -> SkillValidationResult:
    """Validate a Skill object against the Agent Skills specification."""
    result = SkillValidationResult()

    # Name validation
    result.errors.extend(validate_skill_name(skill.name))

    # Name must match parent directory name
    if skill.base_dir and skill.base_dir.name != skill.name:
        result.warnings.append(f"name '{skill.name}' does not match directory name '{skill.base_dir.name}'")

    # Description validation
    if not skill.description:
        result.errors.append("description is required")
    elif len(skill.description) > 1024:
        result.warnings.append(f"description exceeds 1024 characters ({len(skill.description)})")

    # Compatibility validation
    if skill.compatibility and len(skill.compatibility) > 500:
        result.warnings.append(f"compatibility exceeds 500 characters ({len(skill.compatibility)})")

    # Body size recommendation
    if skill.body and len(skill.body.splitlines()) > 500:
        result.warnings.append("body exceeds recommended 500 lines")

    return result


def _parse_allowed_tools(frontmatter: dict) -> Optional[List[str]]:
    """Parse allowed-tools from frontmatter per the Agent Skills spec.

    Spec format: allowed-tools: "Bash(git:*) Read Write"  (space-delimited string)
    Also tolerates YAML array: allowed-tools: [tool1, tool2]
    """
    raw = frontmatter.get("allowed-tools")
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.split() if raw.strip() else None
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return None


def _parse_skill_md(path: Path, source: str = "") -> Optional[Skill]:
    """Parse a SKILL.md file into a Skill object.

    Tolerant parser: loads skills even if they don't strictly follow the spec
    (e.g. Claude Code skills with uppercase names). Validation warnings are
    logged but don't prevent loading.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None

    # Handle BOM
    if content.startswith("\ufeff"):
        content = content[1:]

    # Normalize line endings
    content = content.replace("\r\n", "\n")

    # Split YAML frontmatter from body
    if not content.startswith("---"):
        logger.warning(f"Skill file {path} missing YAML frontmatter (must start with ---)")
        return None

    # Find the closing ---
    try:
        end_idx = content.index("---", 3)
    except ValueError:
        logger.warning(f"Skill file {path} has unclosed YAML frontmatter")
        return None

    frontmatter_str = content[3:end_idx].strip()
    body = content[end_idx + 3 :].strip()

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

    # Coerce to string (some skills use non-string names)
    name = str(name).strip()
    description = str(description).strip() if description else ""

    if not description:
        logger.warning(f"Skill '{name}' at {path} has no description — LLM won't know when to activate it")

    # Parse optional fields
    license_val = frontmatter.get("license")
    compatibility = frontmatter.get("compatibility")
    metadata = frontmatter.get("metadata")
    allowed_tools = _parse_allowed_tools(frontmatter)

    # Coerce types
    if license_val is not None:
        license_val = str(license_val).strip()
    if compatibility is not None:
        compatibility = str(compatibility).strip()
    if metadata is not None and isinstance(metadata, dict):
        metadata = {str(k): str(v) for k, v in metadata.items()}
    else:
        metadata = None

    skill = Skill(
        name=name,
        description=description,
        body=body,
        base_dir=path.parent,
        frontmatter=frontmatter,
        license=license_val,
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=allowed_tools,
        source=source,
    )

    # Validate and log warnings (but still return the skill)
    validation = validate_skill(skill)
    for warning in validation.warnings:
        logger.info(f"Skill '{name}' validation warning: {warning}")
    for error in validation.errors:
        logger.info(f"Skill '{name}' validation issue: {error} (loaded anyway for compatibility)")

    return skill


def _discover_skills_in_dir(skills_dir: Path, source: str = "") -> Dict[str, Skill]:
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

        skill = _parse_skill_md(skill_file, source=source)
        if skill:
            skills[skill.name] = skill
            logger.debug(f"Discovered skill '{skill.name}' at {skill_file} ({source})")

    return skills


# Discovery locations in priority order (lowest to highest).
# Project-local paths are relative to cwd; global paths are absolute.
_GLOBAL_SKILL_DIRS = [
    # (path_resolver, source_label)
    (lambda: get_config_dir() / "skills", "global (cliver)"),
    (lambda: Path.home() / ".agents" / "skills", "global (agents)"),
]

_PROJECT_SKILL_DIRS = [
    # (relative_path, source_label)
    (".cliver/skills", "project (.cliver)"),
    (".agent/skills", "project (.agent)"),
    (".claude/skills", "project (.claude compat)"),
    (".gemini/skills", "project (.gemini compat)"),
    (".qwen/skills", "project (.qwen compat)"),
]

# Builtin skills shipped with the CLIver package (lowest priority).
_BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"


class SkillManager:
    """Discovers, loads, and caches skills from SKILL.md files.

    Skills are discovered from multiple locations in priority order.
    Higher-priority sources override lower-priority ones with the same name.
    """

    def __init__(self):
        self._skills: Optional[Dict[str, Skill]] = None

    def _ensure_loaded(self) -> None:
        if self._skills is not None:
            return
        self._skills = {}

        # 0. Load builtin skills (lowest priority — overridden by everything else)
        try:
            found = _discover_skills_in_dir(_BUILTIN_SKILLS_DIR, source="builtin")
            self._skills.update(found)
        except Exception as e:
            logger.debug(f"Could not scan builtin skills: {e}")

        # 1. Load global skills (lower priority)
        for path_resolver, source in _GLOBAL_SKILL_DIRS:
            try:
                skills_dir = path_resolver()
                found = _discover_skills_in_dir(skills_dir, source=source)
                self._skills.update(found)
            except Exception as e:
                logger.debug(f"Could not scan {source}: {e}")

        # 2. Load project-local skills (higher priority, overrides global)
        cwd = Path.cwd()
        for rel_path, source in _PROJECT_SKILL_DIRS:
            skills_dir = cwd / rel_path
            found = _discover_skills_in_dir(skills_dir, source=source)
            self._skills.update(found)

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
        """Format available skills as a brief, LLM-friendly summary.

        Each skill is listed with its name and a concise description of its goal.
        The list is designed to be quickly scannable so the LLM can decide which
        skill to activate without consuming excessive tokens.
        """
        skills = self.list_skills()
        if not skills:
            return "No skills available."

        lines = [f"Available skills ({len(skills)}):"]
        for skill in skills:
            # Truncate long descriptions to first sentence or 120 chars
            desc = skill.description
            # Take first sentence if description is long
            if len(desc) > 120:
                # Try to cut at first period
                dot = desc.find(". ")
                if 0 < dot <= 120:
                    desc = desc[: dot + 1]
                else:
                    desc = desc[:117] + "..."
            source_tag = f" [{skill.source}]" if skill.source else ""
            lines.append(f"  - {skill.name}: {desc}{source_tag}")

        lines.append("\nCall skill('<name>') to activate a skill and get its full instructions.")
        return "\n".join(lines)

    def activate_skill(self, name: str, prompt: str = "") -> str:
        """Activate a skill and return its content for injection into the conversation.

        Args:
            name: Skill name to activate.
            prompt: Optional initial prompt describing what the user wants.
                    When provided, it is included so the LLM has full context
                    of the task alongside the skill instructions.

        Returns the skill body prefixed with the base directory path and,
        if given, the user's initial prompt.
        """
        skill = self.get_skill(name)
        if not skill:
            available = self.get_skill_names()
            msg = f"Skill '{name}' not found."
            if available:
                msg += f" Available skills: {', '.join(available)}"
            return msg

        parts = [f"Base directory for this skill: {skill.base_dir}/\n\n{skill.body}"]
        if prompt:
            parts.append(f"# User's Request\n\n{prompt}")
        return "\n\n".join(parts)

    def reload(self) -> None:
        """Force re-discovery of skills."""
        self._skills = None
        self._ensure_loaded()


# ---------------------------------------------------------------------------
# Skill execution helpers — used by AgentCore, CLI, and gateway
# ---------------------------------------------------------------------------


def build_skill_appender(
    skills: list[Skill],
    extra_appender: "Callable[[], str] | None" = None,
) -> "Callable[[], str]":
    """Build a system_message_appender that injects skill instructions.

    Combines one or more skill bodies into a single appender. If
    *extra_appender* is provided its output is appended after the skill
    content so callers can layer additional context (e.g. IM rules).
    """
    parts = [f"# Skill: {s.name}\n\nBase directory for this skill: {s.base_dir}/\n\n{s.body}" for s in skills]
    combined = "\n\n".join(parts)

    def _appender() -> str:
        sections = [combined]
        if extra_appender:
            extra = extra_appender()
            if extra:
                sections.append(extra)
        return "\n\n".join(sections)

    return _appender


def build_skill_tool_filter(
    skills: list[Skill],
    extra_filter: "Callable | None" = None,
) -> "Callable | None":
    """Build a filter_tools callback that restricts tools to those listed in
    the skills' ``allowed_tools`` frontmatter.

    Returns *None* if no skill specifies ``allowed_tools`` (meaning no
    filtering is needed). If *extra_filter* is provided it is applied first,
    then the allowed-tools whitelist is applied on top.
    """
    allowed: set[str] = set()
    has_constraint = False
    for s in skills:
        if s.allowed_tools:
            has_constraint = True
            allowed.update(s.allowed_tools)

    if not has_constraint:
        return extra_filter

    async def _filter(user_input: str, tools: list) -> list:
        if extra_filter:
            tools = await extra_filter(user_input, tools)
        return [t for t in tools if t.name in allowed]

    return _filter
