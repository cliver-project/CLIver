"""Skills: frontmatter parsing, spec validation, SkillManager, and run_skill.

The fixture catalogue (``fixtures/skills``) exercises real SKILL.md shapes:
folded descriptions, nested unknown maps, and allowed-tools in both string
and absent forms.
"""

from pathlib import Path

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage, ToolCall
from cliver_core.policy import Decision, ExecutionPolicy
from cliver_core.provider import CLIverResponse
from cliver_core.skill import (
    Skill,
    SkillError,
    SkillManager,
    SkillNotFoundError,
    validate_skill,
    validate_skill_name,
)
from cliver_core.tool import CLIverTool
from cliver_core.tools.run_shell_command import run_shell_command

FIXTURES = Path(__file__).parent / "fixtures" / "skills"


def _make_tool(name: str) -> CLIverTool:
    return CLIverTool(
        name=name,
        description=f"tool {name}",
        parameters={"type": "object", "properties": {}},
        execute=lambda **_kwargs: [{"text": "ok"}],
    )


def _agent(
    tools: list[CLIverTool] | None = None,
    skill_manager: SkillManager | None = None,
    builtin_system_prompt: str | None = None,
) -> AgentCore:
    return AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model")),
        builtin_tools=tools,
        skill_manager=skill_manager,
        builtin_system_prompt=builtin_system_prompt,
    )


def _capture_loop(agent: AgentCore) -> dict:
    captured: dict = {}

    async def fake_run_loop(messages, all_tools, opts, max_iterations, execution_policy=None, skill_state=None):
        captured["messages"] = messages
        captured["tools"] = all_tools
        return None

    agent._run_loop = fake_run_loop  # type: ignore[method-assign]
    return captured


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str | None = None,
    extra_frontmatter: str = "",
    body: str = "Skill instructions.",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    desc = description if description is not None else f"Skill {name} for tests."
    content = f"---\nname: {name}\ndescription: {desc}\n{extra_frontmatter}---\n\n{body}\n"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


# ── Frontmatter parsing ───────────────────────────────────────


def test_parses_folded_literal_nested_and_quoted_frontmatter(tmp_path):
    skill_dir = tmp_path / "kitchen-sink"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: kitchen-sink\n"
        "description: >\n"
        "  Does several things.\n"
        "  Use when tests run.\n"
        'license: "Apache 2.0: the good one"\n'
        "compatibility: |\n"
        "  line one\n"
        "  line two\n"
        "metadata:\n"
        "  author: example-org\n"
        '  version: "1.0"\n'
        "allowed-tools: Read Write\n"
        "unknown-field: kept\n"
        "---\n\n# Body\n\nDo it.\n",
        encoding="utf-8",
    )
    skill = SkillManager(sources=[tmp_path]).get_skill("kitchen-sink")

    assert skill is not None
    assert skill.description == "Does several things. Use when tests run."
    assert skill.license == "Apache 2.0: the good one"
    assert skill.compatibility == "line one\nline two"
    assert skill.metadata == {"author": "example-org", "version": "1.0"}
    assert skill.allowed_tools == ["Read", "Write"]
    assert skill.frontmatter["unknown-field"] == "kept"
    assert skill.body.startswith("# Body")


def test_deeply_nested_cross_agent_metadata(tmp_path):
    """Cross-agent skills (workbuddy, accio, ...) carry nested metadata beyond the spec."""
    skill_dir = tmp_path / "cross-agent"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: cross-agent\n"
        "description: Works in other agents too.\n"
        "metadata:\n"
        "  author: example-org\n"
        "  contact:\n"
        "    name: Team A\n"
        "    email: team@example.com\n"
        '  version: "1.0"\n'
        "x-workbuddy:\n"
        "  enabled: true\n"
        "  triggers:\n"
        "    primary: refund\n"
        "    fallback: |\n"
        "      after-sales\n"
        "      escalation\n"
        "---\n\nBody\n",
        encoding="utf-8",
    )
    skill = SkillManager(sources=[tmp_path]).get_skill("cross-agent")

    assert skill is not None
    assert skill.frontmatter["metadata"]["contact"] == {
        "name": "Team A",
        "email": "team@example.com",
    }
    assert skill.frontmatter["x-workbuddy"]["triggers"]["fallback"] == "after-sales\nescalation"
    # spec-typed field stringifies scalars; nested structure survives as-is
    assert skill.metadata == {
        "author": "example-org",
        "contact": {"name": "Team A", "email": "team@example.com"},
        "version": "1.0",
    }


def test_nested_map_with_sibling_after_deep_block(tmp_path):
    """A sibling key at the parent level ends a nested block, not just same-indent lines."""
    skill_dir = tmp_path / "sibling"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: sibling\n"
        "description: d\n"
        "x-accio:\n"
        "  routing:\n"
        "    intents:\n"
        "      primary: refund\n"
        "  cache: true\n"
        "---\n\nBody\n",
        encoding="utf-8",
    )
    skill = SkillManager(sources=[tmp_path]).get_skill("sibling")

    assert skill is not None
    assert skill.frontmatter["x-accio"] == {
        "routing": {"intents": {"primary": "refund"}},
        "cache": "true",
    }
    assert skill.name == "sibling"  # sibling consumption stopped at the frontmatter fence


def test_list_valued_and_empty_allowed_tools(tmp_path):
    _write_skill(tmp_path, "list-form", extra_frontmatter="allowed-tools: [Read, Write]\n")
    _write_skill(tmp_path, "blank-allow", extra_frontmatter="allowed-tools:\n")
    manager = SkillManager(sources=[tmp_path])

    assert manager.get_skill("list-form").allowed_tools == ["Read", "Write"]
    assert manager.get_skill("blank-allow").allowed_tools is None


def test_malformed_skills_are_skipped(tmp_path):
    no_frontmatter = tmp_path / "no-frontmatter"
    no_frontmatter.mkdir()
    (no_frontmatter / "SKILL.md").write_text("# just a body\n")
    unclosed = tmp_path / "unclosed"
    unclosed.mkdir()
    (unclosed / "SKILL.md").write_text("---\nname: unclosed\n")
    no_name = tmp_path / "no-name"
    no_name.mkdir()
    (no_name / "SKILL.md").write_text("---\ndescription: x\n---\nbody")
    empty_dir = tmp_path / "not-a-skill"  # directory without SKILL.md
    empty_dir.mkdir()
    (tmp_path / "loose-file.md").write_text("ignored")

    assert SkillManager(sources=[tmp_path]).get_skill_names() == []


# ── Spec validation ───────────────────────────────────────────


@pytest.mark.parametrize("name", ["refund", "after-sales", "l4-flow", "a", "a1-b2"])
def test_valid_skill_names(name):
    assert validate_skill_name(name) == []


@pytest.mark.parametrize("name", ["Refund", "-refund", "refund-", "re--fund", "refund!", "x" * 65])
def test_invalid_skill_names(name):
    assert validate_skill_name(name) != []


def test_validate_skill_warnings_do_not_invalidate():
    skill = Skill(
        name="mismatch",
        description="d" * 1025,
        body="",
        base_dir=Path("/skills/other-name"),
    )
    result = validate_skill(skill)

    assert result.is_valid is True
    assert any("directory name" in warning for warning in result.warnings)
    assert any("description exceeds" in warning for warning in result.warnings)


def test_validate_skill_missing_description_is_error():
    skill = Skill(name="ok-name", description="", body="b", base_dir=Path("/skills/ok-name"))
    result = validate_skill(skill)

    assert result.is_valid is False
    assert any("description is required" in error for error in result.errors)


# ── SkillManager ──────────────────────────────────────────────


def test_sources_merge_with_later_override(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _write_skill(base, "alpha", body="base body")
    override = tmp_path / "override"
    override.mkdir()
    _write_skill(override, "alpha", body="override body")
    single = _write_skill(tmp_path, "solo")  # a single skill directory as source

    manager = SkillManager(sources=[base, override, single])

    assert manager.get_skill_names() == ["alpha", "solo"]
    assert manager.get_skill("alpha").body == "override body"
    assert manager.get_skill("solo").source == str(single)


def test_nonexistent_source_is_empty_and_reload_picks_up_new_skills(tmp_path):
    assert SkillManager(sources=[tmp_path / "missing"]).get_skill_names() == []

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "late")
    manager = SkillManager(sources=[skills_dir])
    assert manager.get_skill_names() == ["late"]

    _write_skill(skills_dir, "newcomer")
    assert manager.get_skill_names() == ["late"]  # cached until reload
    manager.reload()
    assert manager.get_skill_names() == ["late", "newcomer"]


def test_format_skill_list_truncates_long_descriptions(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(
        skills_dir,
        "greet",
        description="Greets people in many languages. Use when the user says hello or hi in any "
        "language and wants a greeting back.",
    )

    listing = SkillManager(sources=[skills_dir]).format_skill_list()

    assert "- greet: Greets people in many languages." in listing
    assert "run_skill(" in listing
    assert SkillManager(sources=[]).format_skill_list() == "No skills available."


def test_shorten_description_uses_sentence_boundary_then_cap():
    from cliver_core.skill import _shorten_description

    assert _shorten_description("Short description.") == "Short description."
    assert _shorten_description("First sentence. " + "x" * 200) == "First sentence."
    assert _shorten_description("y" * 200) == "y" * 117 + "..."
    # A boundary within the cap is kept (119 chars here).
    assert _shorten_description("z" * 118 + ". tail") == "z" * 118 + "."
    # A boundary past the cap is not used.
    assert _shorten_description("w" * 130 + ". tail") == "w" * 117 + "..."


# ── Bundled files ─────────────────────────────────────────────


def test_bundled_files_listing_and_reads(tmp_path):
    skill_dir = tmp_path / "bundled"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: bundled\ndescription: d\n---\n\nBody\n")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "policy.md").write_text("# Policy\n\nRefund within 30 days.\n")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("print('hi')\n")

    skill = SkillManager(sources=[skill_dir]).get_skill("bundled")

    assert skill.files == ["references/policy.md", "scripts/run.py"]
    assert "Refund within 30 days." in skill.read_file("references/policy.md")
    with pytest.raises(SkillError, match="escapes"):
        skill.read_file("../outside.md")
    with pytest.raises(SkillError, match="not a file"):
        skill.read_file("references/missing.md")


def test_fixture_skills_load():
    manager = SkillManager(sources=[FIXTURES])

    assert manager.get_skill_names() == ["test-code-review", "test-greeting"]
    greeting = manager.get_skill("test-greeting")
    assert greeting.allowed_tools == ["WebSearch"]
    assert greeting.metadata is None
    assert greeting.frontmatter["parameters"]["default_language"] == "english"
    assert manager.get_skill("test-code-review").allowed_tools is None


# ── AgentCore.run_skill ───────────────────────────────────────


async def test_run_skill_merges_into_single_system_message():
    manager = SkillManager(sources=[FIXTURES])
    agent = _agent(skill_manager=manager, builtin_system_prompt="BASE persona.")
    captured = _capture_loop(agent)

    await agent.run_skill("test-code-review", "check my code", system_prompt="Be terse.")

    messages = captured["messages"]
    assert [m for m in messages if m.role == "system"] == [messages[0]]
    system = messages[0].content
    assert system.startswith("BASE persona.")
    assert "Be terse." in system
    assert "# Active skill: test-code-review" in system
    assert str(FIXTURES / "test-code-review") in system  # bundled scripts need the location
    assert system.index("Be terse.") < system.index("# Active skill:")
    assert "# Code Review Skill" in system


async def test_run_skill_filters_tools_to_allowed_tools():
    manager = SkillManager(sources=[FIXTURES])
    agent = _agent(tools=[_make_tool("WebSearch"), _make_tool("Read")], skill_manager=manager)
    captured = _capture_loop(agent)

    await agent.run_skill("test-greeting", "hello there")

    assert [t.name for t in captured["tools"]] == ["WebSearch"]


async def test_run_skill_without_allowed_tools_advertises_everything():
    manager = SkillManager(sources=[FIXTURES])
    agent = _agent(tools=[_make_tool("WebSearch"), _make_tool("Read")], skill_manager=manager)
    captured = _capture_loop(agent)

    await agent.run_skill("test-code-review", "review please")

    assert sorted(t.name for t in captured["tools"]) == ["Read", "WebSearch"]


async def test_run_skill_unknown_name_lists_available():
    agent = _agent(skill_manager=SkillManager(sources=[FIXTURES]))

    with pytest.raises(SkillNotFoundError, match="test-greeting"):
        await agent.run_skill("nope", "hi")


async def test_run_skill_without_manager_raises():
    agent = _agent()

    with pytest.raises(SkillError, match="SkillManager"):
        await agent.run_skill("test-greeting", "hi")


async def test_run_skill_allowed_tools_unavailable_fails_fast():
    manager = SkillManager(sources=[FIXTURES])
    agent = _agent(tools=[_make_tool("Read")], skill_manager=manager)

    with pytest.raises(SkillError, match="none of which are available"):
        await agent.run_skill("test-greeting", "hi")


async def test_run_skill_accepts_a_loaded_skill_object():
    manager = SkillManager(sources=[FIXTURES])
    agent = _agent(tools=[_make_tool("WebSearch")])  # no manager on the agent
    captured = _capture_loop(agent)
    skill = manager.get_skill("test-greeting")

    await agent.run_skill(skill, "hello")

    assert "# Active skill: test-greeting" in captured["messages"][0].content
    assert [t.name for t in captured["tools"]] == ["WebSearch"]


async def test_run_skill_passes_conversation_through():
    manager = SkillManager(sources=[FIXTURES])
    agent = _agent(skill_manager=manager)
    captured = _capture_loop(agent)
    history = [
        CLIverMessage(role="user", content="earlier"),
        CLIverMessage(role="assistant", content="answer"),
    ]

    await agent.run_skill("test-code-review", "next question", conversation=history)

    messages = captured["messages"]
    assert messages[1:3] == history
    assert messages[-1].content == "next question"


# ── run_skill × execution policy ──────────────────────────────


def _response() -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"), usage=None, media=None)


async def test_run_skill_pins_cwd_with_agent_policy():
    manager = SkillManager(sources=[FIXTURES])
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model")),
        builtin_tools=[run_shell_command],
        skill_manager=manager,
        execution_policy=ExecutionPolicy(),
    )
    captured = {}

    async def fake_loop(messages, tools, opts, max_iterations, execution_policy=None, skill_state=None):
        captured["policy"] = execution_policy
        return _response()

    agent._run_loop = fake_loop

    await agent.run_skill("test-code-review", "review please")

    policy = captured["policy"]
    assert policy is not None
    assert policy.cwd == FIXTURES / "test-code-review"  # agent policy reused, cwd pinned


async def test_run_skill_call_site_policy_verbatim(tmp_path):
    manager = SkillManager(sources=[FIXTURES])
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model")),
        builtin_tools=[run_shell_command],
        skill_manager=manager,
        execution_policy=ExecutionPolicy(cwd=tmp_path),  # would pin elsewhere
    )
    captured = {}

    async def fake_loop(messages, tools, opts, max_iterations, execution_policy=None, skill_state=None):
        captured["policy"] = execution_policy
        return _response()

    agent._run_loop = fake_loop

    call_site = ExecutionPolicy(cwd=FIXTURES)
    await agent.run_skill("test-code-review", "review please", execution_policy=call_site)

    assert captured["policy"] is call_site  # verbatim, not re-pinned


async def test_run_skill_no_policy_pwd_unchanged():
    manager = SkillManager(sources=[FIXTURES])
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model")),
        builtin_tools=[run_shell_command],
        skill_manager=manager,
    )
    captured = {}

    async def fake_loop(messages, tools, opts, max_iterations, execution_policy=None, skill_state=None):
        captured["policy"] = execution_policy
        return _response()

    agent._run_loop = fake_loop

    await agent.run_skill("test-code-review", "review please")

    assert captured["policy"] is None  # opt-in: no agent policy → nothing ambient


async def test_run_skill_allowed_tool_still_denied_by_callback():
    manager = SkillManager(sources=[FIXTURES])

    async def deny(_ctx):
        return Decision.deny("not trusted")

    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model")),
        builtin_tools=[_make_tool("WebSearch")],
        skill_manager=manager,
        permission_callback=deny,
    )
    result = {}

    async def fake_loop(messages, tools, opts, max_iterations, execution_policy=None, skill_state=None):
        web = next(t for t in tools if t.name == "WebSearch")
        result["out"] = await agent._execute_tool(ToolCall(id="c1", name="WebSearch", args={}), web)
        return _response()

    agent._run_loop = fake_loop

    await agent.run_skill("test-greeting", "hello there")

    assert result["out"] == [{"error": "Permission denied: not trusted", "denied": True}]


def test_policy_exports_smoke():
    import cliver_core

    for name in (
        "Decision",
        "DecisionAction",
        "ToolCategory",
        "ToolCallContext",
        "PermissionCallback",
        "EnvSpec",
        "ExecutionPolicy",
        "SAFE_ENV_VARS",
    ):
        assert hasattr(cliver_core, name), name
