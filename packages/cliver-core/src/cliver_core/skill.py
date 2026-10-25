"""Skills — Agent Skills (SKILL.md) loading for AgentCore.

Implements the reading half of the Agent Skills specification
(https://agentskills.io/specification): discovery from explicit sources,
frontmatter parsing, spec validation, and on-demand loading of bundled
files (``scripts/``, ``references/``, ``assets/``).

Deliberately library-shaped: the application decides *where* skills come
from — ``SkillManager(sources=[...])`` takes directories of skills or
single skill directories, and nothing outside a source is ever read.  No
implicit discovery, no environment conventions; those live in the
application layer.

Executing skill scripts and a model-facing file-read tool are future work
(sandbox layer); this module only loads content.

The frontmatter parser is a dependency-free YAML subset — flat scalars,
quoted strings, ``>`` / ``|`` blocks, and nested maps of any depth (the
spec's surface, plus the deeper metadata cross-agent skills carry, e.g.
``x-workbuddy:`` keys).  It is not a general YAML parser by design: no
anchors, no ``-`` sequences.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Skill",
    "SkillError",
    "SkillManager",
    "SkillNotFoundError",
    "validate_skill",
    "validate_skill_name",
]

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"

# Spec: 1-64 chars, lowercase alphanumerics and single hyphens, no
# leading/trailing hyphen, no consecutive hyphens.
_VALID_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){0,63}$")

_BLOCK_INTRO_RE = re.compile(r"^[>|][+-]?$")


class SkillError(Exception):
    """Base class for skill loading and execution errors."""


class SkillNotFoundError(SkillError):
    """A skill named in ``run_skill()`` is not present in the manager."""


# ── Validation ────────────────────────────────────────────────


@dataclass
class SkillValidationResult:
    """Result of validating a skill against the Agent Skills spec."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_skill_name(name: str) -> list[str]:
    """Validate a skill name against the Agent Skills spec (error list; empty = valid)."""
    if not name:
        return ["name is empty"]
    errors: list[str] = []
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
    """Validate a Skill against the Agent Skills specification.

    Name violations and a missing description are errors; length overruns
    and a name/directory mismatch are warnings — the skill stays loadable
    (tolerant loading, for cross-agent compatibility).
    """
    result = SkillValidationResult()

    result.errors.extend(validate_skill_name(skill.name))

    if skill.base_dir and skill.base_dir.name != skill.name:
        result.warnings.append(f"name '{skill.name}' does not match directory name '{skill.base_dir.name}'")

    if not skill.description:
        result.errors.append("description is required")
    elif len(skill.description) > 1024:
        result.warnings.append(f"description exceeds 1024 characters ({len(skill.description)})")

    if skill.compatibility and len(skill.compatibility) > 500:
        result.warnings.append(f"compatibility exceeds 500 characters ({len(skill.compatibility)})")

    if skill.body and len(skill.body.splitlines()) > 500:
        result.warnings.append("body exceeds recommended 500 lines")

    return result


# ── Skill model ───────────────────────────────────────────────


@dataclass
class Skill:
    """A loaded skill parsed from a SKILL.md file."""

    name: str
    description: str
    body: str  # markdown content below the frontmatter
    base_dir: Path  # directory containing SKILL.md

    # Full frontmatter dict (forward-compat with unknown fields)
    frontmatter: dict = field(default_factory=dict)

    # Optional spec fields
    license: str | None = None
    compatibility: str | None = None
    metadata: dict | None = None  # spec: str -> str; nested maps stay structured
    allowed_tools: list[str] | None = None

    # Source label (the path this skill was loaded from)
    source: str = ""

    @property
    def files(self) -> list[str]:
        """Relative paths of every bundled file besides SKILL.md (sorted)."""
        return sorted(
            p.relative_to(self.base_dir).as_posix()
            for p in self.base_dir.rglob("*")
            if p.is_file() and p.name != SKILL_FILENAME
        )

    def read_file(self, relative_path: str, *, max_bytes: int | None = None) -> str:
        """Read one bundled file (``references/``, ``scripts/``, ``assets/``, ...).

        The path must stay inside the skill directory — escaping paths
        (``../``) are rejected.  ``max_bytes`` caps the file size before
        reading (``None`` means no cap).
        """
        root = self.base_dir.resolve()
        target = (self.base_dir / relative_path).resolve()
        if not target.is_relative_to(root):
            raise SkillError(f"{relative_path!r} escapes the skill directory")
        if not target.is_file():
            raise SkillError(f"{relative_path!r} is not a file in skill '{self.name}'")
        if max_bytes is not None and target.stat().st_size > max_bytes:
            raise SkillError(f"{relative_path!r} exceeds {max_bytes} bytes in skill '{self.name}'")
        return target.read_text(encoding="utf-8")


# ── Frontmatter parsing (YAML subset, dependency-free) ────────


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_scalar(raw: str):
    """Parse a scalar value: quoted string, ``[a, b]`` list, or plain text."""
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_strip_quotes(part.strip()) for part in inner.split(",")]
    return _strip_quotes(raw)


def _strip_blank_edges(lines: list[str]) -> list[str]:
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _take_block(lines: list[str], start: int, parent_indent: int) -> tuple[list[str], int]:
    """Collect the block more-indented than *parent_indent* after a ``key:`` line.

    Returns the lines de-indented by the block's first content line (blanks
    preserved inside, stripped at the edges) and the index of the first line
    past the block.  Blank lines between the block and the next sibling key
    belong to neither.
    """
    block: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            block.append("")
            i += 1
            continue
        if len(line) - len(line.lstrip()) <= parent_indent:
            break
        block.append(line)
        i += 1
    block = _strip_blank_edges(block)
    if not block:
        return [], i
    base = len(block[0]) - len(block[0].lstrip())
    return [b[base:] if b else "" for b in block], i


def _fold_block(block: list[str]) -> str:
    """Fold a ``>`` block: single newlines become spaces, blanks become paragraph breaks."""
    text = ""
    for line in block:
        if not line:
            text = text.rstrip(" ") + "\n\n"
        elif not text or text.endswith("\n"):
            text += line
        else:
            text += " " + line
    return text.strip()


def _parse_map_lines(lines: list[str]) -> dict:
    """Parse de-indented ``key: value`` lines into a map, recursing on nesting.

    Cross-agent skills (Claude Code, Gemini, and other ecosystems) add their
    own metadata keys, often nested beyond the spec's one-level ``metadata``
    map — maps here nest to any depth, with scalars, quoted strings, lists,
    and ``>`` / ``|`` blocks behaving exactly like the top level.
    """
    data: dict = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        key, sep, rest = stripped.partition(":")
        i += 1
        if not sep or not key:
            continue
        rest = rest.strip()
        if _BLOCK_INTRO_RE.match(rest):
            block, i = _take_block(lines, i, indent)
            data[key] = _fold_block(block) if rest[0] == ">" else "\n".join(block).strip()
        elif rest:
            data[key] = _parse_scalar(rest)
        else:
            block, i = _take_block(lines, i, indent)
            data[key] = _parse_map_lines(block) if block else None
    return data


def _split_frontmatter(content: str) -> tuple[str, str] | None:
    """Split a SKILL.md into (frontmatter_text, body); None when malformed."""
    if content.startswith("﻿"):
        content = content[1:]
    content = content.replace("\r\n", "\n")
    if not content.startswith("---"):
        return None
    lines = content.split("\n")
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[1:idx]), "\n".join(lines[idx + 1 :]).strip()
    return None


def _parse_frontmatter(text: str) -> dict:
    """Parse the flat YAML-subset frontmatter used by the Agent Skills spec."""
    data: dict = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        i += 1
        if not stripped or stripped.startswith("#"):
            continue
        if line != line.lstrip():
            # Indented line outside any block (orphan) — skip.
            continue
        key, sep, rest = line.partition(":")
        if not sep or not key.strip():
            continue
        key = key.strip()
        rest = rest.strip()
        if _BLOCK_INTRO_RE.match(rest):
            block, i = _take_block(lines, i, 0)
            data[key] = _fold_block(block) if rest[0] == ">" else "\n".join(block).strip()
        elif not rest:
            block, i = _take_block(lines, i, 0)
            data[key] = _parse_map_lines(block) if block else None
        else:
            data[key] = _parse_scalar(rest)
    return data


def _parse_allowed_tools(frontmatter: dict) -> list[str] | None:
    """Parse allowed-tools per spec (space-delimited string); tolerate YAML-style lists."""
    raw = frontmatter.get("allowed-tools")
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.split() if raw.strip() else None
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return None


# ── SKILL.md loading ──────────────────────────────────────────


def _stringify_metadata(mapping: dict) -> dict:
    """Spec says ``metadata`` maps strings to strings; nested maps/lists stay structured."""
    return {
        str(k): v if isinstance(v, (dict, list)) else (str(v) if v is not None else None) for k, v in mapping.items()
    }


def _parse_skill_md(path: Path, source: str = "") -> Skill | None:
    """Parse a SKILL.md file into a Skill.

    Tolerant loader: structural failures (missing frontmatter, missing
    name) log a warning and skip the skill; spec violations only warn.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

    split = _split_frontmatter(content)
    if split is None:
        logger.warning("Skill file %s has missing or unclosed YAML frontmatter", path)
        return None
    frontmatter_text, body = split
    frontmatter = _parse_frontmatter(frontmatter_text)

    name = frontmatter.get("name")
    if not name or not str(name).strip():
        logger.warning("Skill file %s is missing the required 'name' field", path)
        return None
    name = str(name).strip()
    description = str(frontmatter.get("description") or "").strip()

    license_value = frontmatter.get("license")
    compatibility = frontmatter.get("compatibility")
    metadata = frontmatter.get("metadata")
    skill = Skill(
        name=name,
        description=description,
        body=body,
        base_dir=path.parent,
        frontmatter=frontmatter,
        license=str(license_value).strip() if license_value is not None else None,
        compatibility=str(compatibility).strip() if compatibility is not None else None,
        metadata=_stringify_metadata(metadata) if isinstance(metadata, dict) else None,
        allowed_tools=_parse_allowed_tools(frontmatter),
        source=source,
    )

    validation = validate_skill(skill)
    for warning in validation.warnings:
        logger.info("Skill '%s' validation warning: %s", name, warning)
    for error in validation.errors:
        logger.info("Skill '%s' validation issue: %s (loaded anyway for compatibility)", name, error)
    if not skill.description:
        logger.warning("Skill '%s' has no description — the model won't know when to use it", name)
    return skill


def _discover_source(source_path: Path, source: str) -> dict[str, Skill]:
    """Load skills from one source path.

    A directory that contains SKILL.md is a single skill; any other
    directory is a container scanned one level deep for skill directories.
    """
    if not source_path.is_dir():
        return {}
    if (source_path / SKILL_FILENAME).is_file():
        skill = _parse_skill_md(source_path / SKILL_FILENAME, source=source)
        return {skill.name: skill} if skill else {}
    skills: dict[str, Skill] = {}
    for child in sorted(source_path.iterdir()):
        if not child.is_dir():
            continue
        skill_file = child / SKILL_FILENAME
        if not skill_file.is_file():
            continue
        skill = _parse_skill_md(skill_file, source=source)
        if skill:
            skills[skill.name] = skill
    return skills


# ── Display helpers ───────────────────────────────────────────


_DESCRIPTION_MAX_CHARS = 120


def _shorten_description(description: str) -> str:
    """Shorten a skill description for prompt/listing use.

    Prefers a sentence boundary within the cap, else hard-truncates.
    """
    if len(description) <= _DESCRIPTION_MAX_CHARS:
        return description
    dot = description.find(". ")
    if 0 < dot <= _DESCRIPTION_MAX_CHARS:
        return description[: dot + 1]
    return description[: _DESCRIPTION_MAX_CHARS - 3] + "..."


# ── SkillManager ──────────────────────────────────────────────


class SkillManager:
    """Loads and caches Agent Skills from explicit application-supplied sources.

    Each source is a directory of skill directories or a single skill
    directory.  Later sources override earlier ones with the same skill
    name, so put overrides last.  Sources are read lazily on first access
    (thread-safe, single load) and re-read after :meth:`reload`.
    """

    def __init__(self, sources: list[str | Path] | None = None):
        self._sources = [(Path(p).expanduser(), str(p)) for p in (sources or [])]
        self._skills: dict[str, Skill] | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> dict[str, Skill]:
        if self._skills is not None:
            return self._skills
        with self._lock:
            if self._skills is not None:
                return self._skills
            skills: dict[str, Skill] = {}
            for source_path, label in self._sources:
                skills.update(_discover_source(source_path, label))
            self._skills = skills
            if skills:
                logger.info("Loaded %d skill(s): %s", len(skills), sorted(skills))
            else:
                logger.debug("No skills found in %d source(s)", len(self._sources))
            return skills

    def list_skills(self) -> list[Skill]:
        """All discovered skills (routing metadata level — name + description)."""
        return list(self._ensure_loaded().values())

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name, or None."""
        return self._ensure_loaded().get(name)

    def get_skill_names(self) -> list[str]:
        """All available skill names."""
        return list(self._ensure_loaded())

    def format_skill_list(self) -> str:
        """Format the catalogue as a compact, model-scannable summary."""
        skills = self.list_skills()
        if not skills:
            return "No skills available."
        lines = [f"Available skills ({len(skills)}):"]
        for skill in skills:
            lines.append(f"  - {skill.name}: {_shorten_description(skill.description)}")
        lines.append("\nActivate a skill with run_skill('<name>') to run it.")
        return "\n".join(lines)

    def reload(self) -> None:
        """Drop the cache; sources are re-read on next access."""
        with self._lock:
            self._skills = None
