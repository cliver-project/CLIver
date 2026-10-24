"""Model-driven skill dispatch: catalogue prompt, FindSkills, ActivateSkill.

Covers the self-contained ``skill_tools`` module (formatting, default search,
tool factories, run state).  AgentCore wiring lives in ``test_agent_skills.py``.
"""

from pathlib import Path

from cliver_core.skill import SkillManager
from cliver_core.skill_tools import (
    ACTIVATE_SKILL_TOOL,
    DEFAULT_SEARCH_LIMIT,
    FIND_SKILLS_TOOL,
    MAX_SEARCH_LIMIT,
    SkillRunState,
    format_skill_body,
    format_skill_catalog,
    lexical_skill_search,
    make_activate_skill_tool,
    make_find_skills_tool,
)


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "",
    extra_frontmatter: str = "",
    body: str = "Skill instructions.",
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    desc = description or f"Skill {name} for tests."
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n{extra_frontmatter}---\n\n{body}\n",
        encoding="utf-8",
    )


def _manager(tmp_path: Path, *, extra_frontmatter: dict[str, str] | None = None) -> SkillManager:
    root = tmp_path / "skills"
    root.mkdir()
    frontmatter = extra_frontmatter or {}
    _write_skill(
        root,
        "alpha",
        description="Computes weather statistics from forecast data.",
        extra_frontmatter=frontmatter.get("alpha", ""),
        body="# Alpha\n\nAlpha instructions.",
    )
    _write_skill(
        root,
        "beta",
        description="Writes haiku poems about the seasons.",
        extra_frontmatter=frontmatter.get("beta", ""),
        body="# Beta\n\nBeta instructions.",
    )
    _write_skill(root, "gamma", description="Builds weather forecast summaries.", body="# Gamma\n\nGamma body.")
    return SkillManager(sources=[root])


# ── Catalogue prompt ──────────────────────────────────────────


def test_catalog_lists_selected_skills_and_tool_hints(tmp_path):
    manager = _manager(tmp_path)

    text = format_skill_catalog(manager.list_skills()[:2], total=5)

    assert text.startswith("## Skills")
    assert "2 of 5" in text
    assert FIND_SKILLS_TOOL in text and ACTIVATE_SKILL_TOOL in text
    assert "- alpha: Computes weather statistics from forecast data." in text
    assert "- beta: Writes haiku poems about the seasons." in text
    assert "gamma" not in text


def test_catalog_states_tool_precedence_over_skills(tmp_path):
    manager = _manager(tmp_path)

    text = format_skill_catalog(manager.list_skills()[:2], total=5)

    assert "Tools are the primary way to act" in text
    assert "use the tool instead of a skill" in text


def test_activate_skill_description_requires_tool_precedence(tmp_path):
    manager = _manager(tmp_path)

    activate = make_activate_skill_tool(manager, SkillRunState(), set())

    assert "no available tool can do the task" in activate.description


def test_catalog_shortens_long_descriptions(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    _write_skill(
        root,
        "long",
        description="First sentence of the description. " + "padding " * 40,
    )
    skill = SkillManager(sources=[root]).get_skill("long")

    text = format_skill_catalog([skill], total=1)

    assert "- long: First sentence of the description." in text
    assert "padding padding padding" not in text


def test_catalog_empty_selection_keeps_header_and_hint(tmp_path):
    text = format_skill_catalog([], total=3)

    assert "## Skills" in text
    assert "3 available" in text
    assert FIND_SKILLS_TOOL in text and ACTIVATE_SKILL_TOOL in text
    assert "\n- " not in text


def test_format_skill_body_matches_run_skill_framing(tmp_path):
    skill = _manager(tmp_path).get_skill("alpha")

    text = format_skill_body(skill)

    assert text.startswith("# Active skill: alpha")
    assert f"Skill location: {skill.base_dir}" in text
    assert "# Alpha" in text
    assert "Alpha instructions." in text


# ── Default lexical search ───────────────────────────────────


def test_lexical_search_matches_name_and_description(tmp_path):
    manager = _manager(tmp_path)
    skills = manager.list_skills()

    assert [s.name for s in lexical_skill_search("weather", skills)] == ["alpha", "gamma"]
    assert [s.name for s in lexical_skill_search("HAIKU", skills)] == ["beta"]
    assert [s.name for s in lexical_skill_search("zzz", skills)] == []


def test_lexical_search_empty_query_lists_all_sorted_by_name(tmp_path):
    skills = _manager(tmp_path).list_skills()

    assert [s.name for s in lexical_skill_search("  ", skills)] == ["alpha", "beta", "gamma"]


# ── FindSkills tool ──────────────────────────────────────────


def test_find_skills_tool_default_limit_and_cap(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for index in range(MAX_SEARCH_LIMIT + 5):
        _write_skill(root, f"data-{index:02d}", description="Processes data files.")
    manager = SkillManager(sources=[root])

    tool = make_find_skills_tool(manager)
    default = tool.execute(query="data")[0]["text"]
    capped = tool.execute(query="data", limit=999)[0]["text"]

    assert default.count("\n- ") + default.startswith("- ") == DEFAULT_SEARCH_LIMIT
    assert capped.count("\n- ") + capped.startswith("- ") == MAX_SEARCH_LIMIT


def test_find_skills_tool_no_match_lists_available_names(tmp_path):
    tool = make_find_skills_tool(_manager(tmp_path))

    result = tool.execute(query="zzz")

    assert "No skills matched" in result[0]["text"]
    assert "alpha" in result[0]["text"] and "gamma" in result[0]["text"]


def test_find_skills_tool_uses_custom_search(tmp_path):
    manager = _manager(tmp_path)

    def search(query, skills):
        assert query == "anything"
        return [s for s in skills if s.name == "beta"]

    tool = make_find_skills_tool(manager, search)

    text = tool.execute(query="anything")[0]["text"]
    assert "- beta: Writes haiku poems about the seasons." in text
    assert "alpha" not in text


def test_find_skills_tool_awaits_an_async_custom_search(tmp_path):
    manager = _manager(tmp_path)

    async def search(query, skills):
        return [s for s in skills if s.name == "gamma"]

    tool = make_find_skills_tool(manager, search)

    assert "- gamma:" in tool.execute(query="x")[0]["text"]


# ── ActivateSkill tool ───────────────────────────────────────


def test_skill_run_state_starts_idle():
    state = SkillRunState()

    assert state.active is None
    assert state.narrowed is False


def test_activate_skill_returns_body_and_marks_state(tmp_path):
    manager = _manager(tmp_path)
    state = SkillRunState()
    tool = make_activate_skill_tool(manager, state, available_tool_names={"Read"})

    result = tool.execute(name="alpha")

    assert state.active is not None and state.active.name == "alpha"
    assert result[0]["text"] == format_skill_body(manager.get_skill("alpha"))


def test_activate_skill_unknown_name_lists_available(tmp_path):
    state = SkillRunState()
    tool = make_activate_skill_tool(_manager(tmp_path), state, available_tool_names=set())

    result = tool.execute(name="nope")

    assert "not found" in result[0]["error"]
    assert "alpha" in result[0]["error"]
    assert state.active is None


def test_activate_skill_rejects_a_second_activation(tmp_path):
    manager = _manager(tmp_path)
    state = SkillRunState()
    tool = make_activate_skill_tool(manager, state, available_tool_names={"Read"})
    tool.execute(name="alpha")

    result = tool.execute(name="beta")

    assert "already active" in result[0]["error"]
    assert state.active.name == "alpha"


def test_activate_skill_rejects_unavailable_allowlist(tmp_path):
    manager = _manager(
        tmp_path,
        extra_frontmatter={"alpha": "allowed-tools: WebSearch Bash\n"},
    )
    state = SkillRunState()
    tool = make_activate_skill_tool(manager, state, available_tool_names={"Read"})

    result = tool.execute(name="alpha")

    assert "none of which are available" in result[0]["error"]
    assert "WebSearch" in result[0]["error"] and "Bash" in result[0]["error"]
    assert state.active is None


def test_activate_skill_empty_allowlist_behaves_like_absent(tmp_path):
    manager = _manager(tmp_path, extra_frontmatter={"beta": "allowed-tools: []\n"})
    state = SkillRunState()
    tool = make_activate_skill_tool(manager, state, available_tool_names=set())

    result = tool.execute(name="beta")

    assert "error" not in result[0]
    assert state.active.name == "beta"
