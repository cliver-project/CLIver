"""Model-driven skill dispatch — catalogue prompt and skill tools.

Progressive disclosure for Agent Skills: a short catalogue in the system
prompt (which skills exist), ``FindSkills`` to search the full catalogue,
and ``ActivateSkill`` to load one skill's body as a tool result.

The application owns *which* skills exist (``SkillManager`` sources) and may
replace the summary ranking (``SkillSelector``) or the search itself
(``SkillSearch``) — e.g. with an encoder-only / system-1 model.  Both seams
accept sync or async callables; core ships dependency-free defaults
(alphabetical summary, lexical search), so the feature works with no app
code at all.

The tools are built per AgentCore/run and are not part of the builtin tool
package: they need the agent's ``SkillManager`` and per-run activation state.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from cliver_core.skill import Skill, SkillManager, _shorten_description
from cliver_core.tool import CLIverTool, tool

__all__ = [
    "ACTIVATE_SKILL_TOOL",
    "DEFAULT_SEARCH_LIMIT",
    "FIND_SKILLS_TOOL",
    "MAX_SEARCH_LIMIT",
    "SKILL_TOOL_NAMES",
    "SkillRunState",
    "SkillSearch",
    "SkillSelector",
    "format_skill_body",
    "format_skill_catalog",
    "lexical_skill_search",
    "make_activate_skill_tool",
    "make_find_skills_tool",
]

FIND_SKILLS_TOOL = "FindSkills"
ACTIVATE_SKILL_TOOL = "ActivateSkill"
SKILL_TOOL_NAMES = frozenset({FIND_SKILLS_TOOL, ACTIVATE_SKILL_TOOL})

DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 20

# App seams: rank the prompt summary (popularity today, encoder/system-1
# relevance later) and search the catalogue (semantic retrieval later).
# The selector receives the raw user input; the search receives the model's
# query.  Both may return an awaitable.
SkillSelector = Callable[[Sequence[Skill], str], Sequence[Skill] | Awaitable[Sequence[Skill]]]
SkillSearch = Callable[[str, Sequence[Skill]], Sequence[Skill] | Awaitable[Sequence[Skill]]]


@dataclass
class SkillRunState:
    """Mutable skill state for one ``chat()`` / ``stream()`` run.

    ``active`` is the skill the model activated; ``narrowed`` records that
    the run's advertised tools were already restricted to its
    ``allowed-tools``.
    """

    active: Skill | None = None
    narrowed: bool = False


# ── Prompt formatting ─────────────────────────────────────────


def format_skill_catalog(skills: Sequence[Skill], total: int) -> str:
    """Render the system-prompt skill summary for *skills* out of *total*.

    *skills* is the (possibly empty) selected subset; *total* is the full
    catalogue size, so the model knows the list is partial.
    """
    listed = len(skills)
    if listed:
        summary = f"This summary lists {listed} of {total} available skill(s)."
    else:
        summary = f"The summary currently lists no skills ({total} available)."
    lines = [
        "## Skills",
        "",
        "Skills are reusable instruction packs for specific tasks, separate from tools. "
        "Tools are the primary way to act: if an available tool can do the task, use the tool "
        "instead of a skill. Activate a skill only when no tool can do the task, or when the "
        f"task needs the skill's specific procedure. {summary} Call {FIND_SKILLS_TOOL}(query=...) "
        f"to search the full catalogue (an empty query lists every name), and "
        f"{ACTIVATE_SKILL_TOOL}(name=...) to load a skill's full instructions when it is the "
        "right choice.",
    ]
    if listed:
        lines.append("")
        for skill in skills:
            lines.append(f"- {skill.name}: {_shorten_description(skill.description)}")
    return "\n".join(lines)


def format_skill_body(skill: Skill) -> str:
    """Frame a skill body for injection (shared by run_skill and ActivateSkill).

    The location line is required for skills that bundle scripts: the model
    cannot run ``scripts/*.py`` without knowing where they live.
    """
    return f"# Active skill: {skill.name}\n\nSkill location: {skill.base_dir}\n\n{skill.body}"


# ── Search ────────────────────────────────────────────────────


def lexical_skill_search(query: str, skills: Sequence[Skill]) -> list[Skill]:
    """Default catalogue search: case-insensitive tokens over name + description.

    Every matching token adds to a skill's score; results are ordered by
    score (descending) then name.  An empty query lists every skill by name.
    """
    terms = query.lower().split()
    if not terms:
        return sorted(skills, key=lambda skill: skill.name)
    scored: list[tuple[int, str, Skill]] = []
    for skill in skills:
        haystack = f"{skill.name} {skill.description}".lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, skill.name, skill))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [skill for _, _, skill in scored]


def _resolve(value):
    """Return *value*, driving it with a private event loop when it is awaitable.

    ``FindSkills`` runs in a worker thread (``asyncio.to_thread``), so an
    async app search — e.g. encoder-based retrieval — can be driven there.
    """
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


# ── Tools ─────────────────────────────────────────────────────

_FIND_SKILLS_DESCRIPTION = (
    "Search available skills by keyword. A skill is a reusable instruction pack for a "
    "specific task, separate from tools. Use this to discover skills the system prompt "
    "does not list: pass keywords from the task, or an empty query to list every skill. "
    f"Returns up to limit matches (default {DEFAULT_SEARCH_LIMIT}, max {MAX_SEARCH_LIMIT}) "
    f"with names and descriptions; load one with {ACTIVATE_SKILL_TOOL}(name=...)."
)

_ACTIVATE_SKILL_DESCRIPTION = (
    "Load a skill's full instructions into the conversation, then follow them. Activate a "
    "skill only when no available tool can do the task, or when the task needs the skill's "
    "specific procedure; otherwise use the tools. Pick a skill from the system-prompt summary "
    f"or {FIND_SKILLS_TOOL}(query=...). After activation the skill's allowed-tools restrict "
    "which tools are available for the rest of the run; only one skill can be active per run."
)


def make_find_skills_tool(manager: SkillManager, search: SkillSearch | None = None) -> CLIverTool:
    """Build the ``FindSkills`` tool bound to *manager* and an optional search seam."""

    @tool(name=FIND_SKILLS_TOOL, description=_FIND_SKILLS_DESCRIPTION, safety="safe")
    def find_skills(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        """Search the skill catalogue.

        Args:
            query: Keywords matched against skill names and descriptions. An
                empty query lists every skill.
            limit: Maximum number of matches to return (default 5, max 20).
        """
        skills = manager.list_skills()
        if not skills:
            return [{"text": "No skills are available."}]
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        if search is not None:
            matches = list(_resolve(search(query, skills)))
        else:
            matches = lexical_skill_search(query, skills)
        if not matches:
            names = ", ".join(skill.name for skill in sorted(skills, key=lambda s: s.name))
            return [{"text": f"No skills matched {query!r}. Available skills: {names}."}]
        shown = matches[:limit]
        header = f"Matched {len(shown)} of {len(matches)} skill(s) (call {ACTIVATE_SKILL_TOOL}(name=...) to load one):"
        lines = [header]
        lines.extend(f"- {s.name}: {_shorten_description(s.description)}" for s in shown)
        return [{"text": "\n".join(lines)}]

    return find_skills


def make_activate_skill_tool(
    manager: SkillManager,
    state: SkillRunState,
    available_tool_names: set[str],
) -> CLIverTool:
    """Build the ``ActivateSkill`` tool for one run.

    *available_tool_names* is the run's gathered regular tools (skill tools
    excluded); it backs the fail-fast check when a skill's ``allowed-tools``
    matches nothing the run can offer.
    """

    @tool(name=ACTIVATE_SKILL_TOOL, description=_ACTIVATE_SKILL_DESCRIPTION, safety="safe")
    def activate_skill(name: str) -> list[dict]:
        """Activate one skill for the rest of the run.

        Args:
            name: The skill name (from the system-prompt summary, FindSkills,
                or the user).
        """
        if state.active is not None:
            return [
                {
                    "error": f"skill '{state.active.name}' is already active in this run; "
                    "only one skill can be activated per run"
                }
            ]
        skill = manager.get_skill(name)
        if skill is None:
            available = ", ".join(manager.get_skill_names()) or "(none)"
            return [{"error": f"skill '{name}' not found. Available skills: {available}"}]
        allowed = frozenset(skill.allowed_tools or ())
        if allowed and not (allowed & available_tool_names):
            return [
                {
                    "error": f"skill '{name}' allows only tools {sorted(allowed)}, none of which "
                    f"are available (available: {sorted(available_tool_names)}). Register the "
                    "tools or connect the MCP servers the skill needs."
                }
            ]
        state.active = skill
        return [{"text": format_skill_body(skill)}]

    return activate_skill
