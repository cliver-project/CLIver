"""AgentCore wiring for model-driven skill dispatch (catalogue + tools + narrowing)."""

from pathlib import Path

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.skill import SkillManager
from cliver_core.tool import CLIverTool

FIXTURES = Path(__file__).parent / "fixtures" / "skills"


def _manager() -> SkillManager:
    return SkillManager(sources=[FIXTURES])


def _make_tool(name: str) -> CLIverTool:
    return CLIverTool(
        name=name,
        description=f"tool {name}",
        parameters={"type": "object", "properties": {}},
        execute=lambda **_kwargs: [{"text": "ok"}],
    )


def _agent(**kwargs) -> AgentCore:
    return AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model")), **kwargs)


def _capture_loop(agent: AgentCore) -> dict:
    captured: dict = {}

    async def fake_run_loop(messages, all_tools, opts, max_iterations, execution_policy=None, skill_state=None):
        captured["messages"] = messages
        captured["tools"] = all_tools
        captured["skill_state"] = skill_state
        return None

    agent._run_loop = fake_run_loop  # type: ignore[method-assign]
    return captured


def _system(captured: dict) -> str:
    return captured["messages"][0].content


def _tool_names(captured: dict) -> list[str]:
    return [tool.name for tool in captured["tools"]]


# ── Catalogue prompt ─────────────────────────────────────────


async def test_catalog_section_follows_caller_prompt():
    agent = _agent(skill_manager=_manager(), builtin_system_prompt="BASE persona.")
    captured = _capture_loop(agent)

    await agent.chat("review this code", system_prompt="Be terse.")

    system = _system(captured)
    assert system.startswith("BASE persona.")
    assert system.index("Be terse.") < system.index("## Skills")
    assert "2 of 2" in system
    assert "- test-code-review:" in system
    assert "- test-greeting:" in system
    assert "FindSkills" in system and "ActivateSkill" in system


async def test_selector_receives_user_input_and_controls_order_and_limit():
    seen: dict = {}

    def selector(skills, user_input):
        seen["names"] = [skill.name for skill in skills]
        seen["input"] = user_input
        return list(reversed(skills))

    agent = _agent(skill_manager=_manager(), skill_selector=selector, skill_summary_limit=1)
    captured = _capture_loop(agent)

    await agent.chat("hello there")

    assert seen["input"] == "hello there"
    assert seen["names"] == ["test-code-review", "test-greeting"]
    system = _system(captured)
    assert "1 of 2" in system
    assert "- test-greeting:" in system
    assert "- test-code-review:" not in system


async def test_async_selector_is_awaited():
    async def selector(skills, user_input):
        return [skill for skill in skills if skill.name == "test-greeting"]

    agent = _agent(skill_manager=_manager(), skill_selector=selector)
    captured = _capture_loop(agent)

    await agent.chat("hi")

    system = _system(captured)
    assert "- test-greeting:" in system
    assert "- test-code-review:" not in system


async def test_no_catalog_without_a_manager():
    agent = _agent()
    captured = _capture_loop(agent)

    await agent.chat("hi")

    assert "## Skills" not in _system(captured)


async def test_empty_selection_keeps_header_and_hint():
    agent = _agent(skill_manager=_manager(), skill_selector=lambda skills, user_input: [])
    captured = _capture_loop(agent)

    await agent.chat("hi")

    system = _system(captured)
    assert "## Skills" in system
    assert "no skills (2 available)" in system
    assert "FindSkills" in system


async def test_zero_limit_hides_the_list_but_keeps_the_dispatch_tools():
    agent = _agent(skill_manager=_manager(), skill_summary_limit=0)
    captured = _capture_loop(agent)

    await agent.chat("hi")

    system = _system(captured)
    assert "no skills (2 available)" in system
    assert "- test-greeting:" not in system
    assert _tool_names(captured) == ["FindSkills", "ActivateSkill"]


def test_negative_summary_limit_raises():
    with pytest.raises(ValueError, match="skill_summary_limit"):
        _agent(skill_manager=_manager(), skill_summary_limit=-1)


# ── Advertised tools ─────────────────────────────────────────


async def test_skill_tools_are_advertised_when_dispatch_is_on():
    agent = _agent(skill_manager=_manager(), builtin_tools=[_make_tool("Read")])
    captured = _capture_loop(agent)

    await agent.chat("hi")

    assert _tool_names(captured) == ["Read", "FindSkills", "ActivateSkill"]


async def test_tool_filter_applies_only_to_regular_tools():
    agent = _agent(skill_manager=_manager(), builtin_tools=[_make_tool("Read")])
    captured = _capture_loop(agent)

    await agent.chat("hi", tool_filter=lambda _tool: False)

    assert _tool_names(captured) == ["FindSkills", "ActivateSkill"]


async def test_skill_tool_name_collision_raises():
    agent = _agent(skill_manager=_manager(), builtin_tools=[_make_tool("FindSkills")])

    with pytest.raises(ValueError, match="reserved"):
        await agent.chat("hi")


async def test_skill_dispatch_false_disables_catalog_and_tools():
    agent = _agent(skill_manager=_manager(), builtin_tools=[_make_tool("Read")])
    captured = _capture_loop(agent)

    await agent.chat("hi", skill_dispatch=False)

    assert "## Skills" not in _system(captured)
    assert _tool_names(captured) == ["Read"]


async def test_run_skill_disables_dispatch():
    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("WebSearch")],
    )
    captured = _capture_loop(agent)

    await agent.run_skill("test-code-review", "review please")

    system = _system(captured)
    assert "# Active skill: test-code-review" in system
    assert "## Skills" not in system
    assert _tool_names(captured) == ["Read", "WebSearch"]


# ── Tool candidate guardrail ─────────────────────────────────


async def test_tool_selector_receives_user_input_and_filters_candidates():
    seen: dict = {}

    def selector(tools, user_input):
        seen["names"] = [tool.name for tool in tools]
        seen["input"] = user_input
        return [tool for tool in tools if tool.name == "Read"]

    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("Bash")],
        tool_selector=selector,
    )
    captured = _capture_loop(agent)

    await agent.chat("read the file")

    assert seen["input"] == "read the file"
    assert seen["names"] == ["Bash", "Read"]  # registry order (sorted by name)
    assert _tool_names(captured) == ["Read", "FindSkills", "ActivateSkill"]


async def test_async_tool_selector_is_awaited():
    async def selector(tools, user_input):
        return [tool for tool in tools if tool.name == "Bash"]

    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("Bash")],
        tool_selector=selector,
    )
    captured = _capture_loop(agent)

    await agent.chat("run the tests")

    assert _tool_names(captured) == ["Bash", "FindSkills", "ActivateSkill"]


async def test_tool_selector_runs_over_extra_tools():
    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read")],
        tool_selector=lambda tools, user_input: [tool for tool in tools if tool.name != "Read"],
    )
    captured = _capture_loop(agent)

    await agent.chat("hi", extra_tools=[_make_tool("Extra")])

    assert _tool_names(captured) == ["Extra", "FindSkills", "ActivateSkill"]


async def test_tool_selector_cannot_remove_skill_dispatch_tools():
    agent = _agent(skill_manager=_manager(), tool_selector=lambda tools, user_input: [])
    captured = _capture_loop(agent)

    await agent.chat("hi")

    assert _tool_names(captured) == ["FindSkills", "ActivateSkill"]


async def test_tool_selector_composes_with_call_tool_filter():
    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("Bash")],
        tool_selector=lambda tools, user_input: [tool for tool in tools if tool.name == "Read"],
    )
    captured = _capture_loop(agent)

    await agent.chat("hi", tool_filter=lambda tool: tool.name != "Read")

    assert _tool_names(captured) == ["FindSkills", "ActivateSkill"]


async def test_stream_applies_tool_selector():
    provider = _ActivatingStreamProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[_make_tool("Read"), _make_tool("WebSearch")],
        skill_manager=_manager(),
        tool_selector=lambda tools, user_input: [tool for tool in tools if tool.name == "WebSearch"],
    )

    [chunk async for chunk in agent.stream("greet me")]

    assert [tool.name for tool in provider.requests[0].tools] == [
        "WebSearch",
        "FindSkills",
        "ActivateSkill",
    ]


# ── Cross-category candidate guardrail ───────────────────────


async def test_candidate_selector_sees_tools_and_skills_and_filters_both():
    seen: dict = {}

    def selector(tools, skills, user_input):
        seen["tools"] = [tool.name for tool in tools]
        seen["skills"] = [skill.name for skill in skills]
        seen["input"] = user_input
        return (
            [tool for tool in tools if tool.name == "Read"],
            [skill for skill in skills if skill.name == "test-greeting"],
        )

    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("Bash")],
        candidate_selector=selector,
    )
    captured = _capture_loop(agent)

    await agent.chat("greet me")

    assert seen["input"] == "greet me"
    assert seen["tools"] == ["Bash", "Read"]
    assert seen["skills"] == ["test-code-review", "test-greeting"]
    assert _tool_names(captured) == ["Read", "FindSkills", "ActivateSkill"]
    system = _system(captured)
    assert "1 of 2" in system
    assert "- test-greeting:" in system
    assert "- test-code-review:" not in system


async def test_async_candidate_selector_is_awaited():
    async def selector(tools, skills, user_input):
        return [tool for tool in tools if tool.name == "Bash"], skills

    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("Bash")],
        candidate_selector=selector,
    )
    captured = _capture_loop(agent)

    await agent.chat("run the tests")

    assert _tool_names(captured) == ["Bash", "FindSkills", "ActivateSkill"]


async def test_candidate_selector_sees_extra_tools():
    def selector(tools, skills, user_input):
        return [tool for tool in tools if tool.name == "Extra"], skills

    agent = _agent(skill_manager=_manager(), candidate_selector=selector)
    captured = _capture_loop(agent)

    await agent.chat("hi", extra_tools=[_make_tool("Extra")])

    assert _tool_names(captured) == ["Extra", "FindSkills", "ActivateSkill"]


async def test_candidate_selector_composes_with_individual_selectors():
    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("Bash")],
        candidate_selector=lambda tools, skills, user_input: (tools, [s for s in skills if s.name == "test-greeting"]),
        tool_selector=lambda tools, user_input: [tool for tool in tools if tool.name != "Read"],
        skill_selector=lambda skills, user_input: [],
    )
    captured = _capture_loop(agent)

    await agent.chat("hi")

    assert _tool_names(captured) == ["Bash", "FindSkills", "ActivateSkill"]
    assert "no skills (2 available)" in _system(captured)


async def test_candidate_selector_receives_no_skills_when_dispatch_is_off():
    seen: dict = {}

    def selector(tools, skills, user_input):
        seen["skills"] = list(skills)
        return [tool for tool in tools if tool.name == "WebSearch"], skills

    agent = _agent(
        skill_manager=_manager(),
        builtin_tools=[_make_tool("Read"), _make_tool("WebSearch")],
        candidate_selector=selector,
    )
    captured = _capture_loop(agent)

    await agent.run_skill("test-code-review", "review please")

    assert seen["skills"] == []
    assert _tool_names(captured) == ["WebSearch"]


async def test_stream_applies_candidate_selector():
    provider = _ActivatingStreamProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[_make_tool("Read"), _make_tool("WebSearch")],
        skill_manager=_manager(),
        candidate_selector=lambda tools, skills, user_input: (
            [tool for tool in tools if tool.name == "WebSearch"],
            skills,
        ),
    )

    [chunk async for chunk in agent.stream("greet me")]

    assert [tool.name for tool in provider.requests[0].tools] == [
        "WebSearch",
        "FindSkills",
        "ActivateSkill",
    ]


# ── Activation narrows the run ───────────────────────────────


class _ActivatingProvider(Provider):
    """Turn 1 activates a skill (default ``test-greeting``, allowlist: WebSearch); turn 2 answers."""

    protocol = "openai"
    brand = "mock-activate"
    skill_name = "test-greeting"

    def __init__(self, skill_name: str | None = None) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []
        if skill_name is not None:
            self.skill_name = skill_name

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="ActivateSkill", args={"name": self.skill_name})],
                ),
                usage=None,
                media=None,
            )
        return CLIverResponse(message=CLIverMessage(role="assistant", content="done"), usage=None, media=None)

    async def stream(self, request):  # pragma: no cover — chat test only
        if False:
            yield None


class _ReadSkillFileProvider(Provider):
    """Turn 1 activates test-greeting; turn 2 reads a bundled reference file; turn 3 answers."""

    protocol = "openai"
    brand = "mock-read-skill-file"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="ActivateSkill", args={"name": "test-greeting"})],
                ),
                usage=None,
                media=None,
            )
        if len(self.requests) == 2:
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="ReadSkillFile",
                            args={"path": "references/phrases.md"},
                        )
                    ],
                ),
                usage=None,
                media=None,
            )
        return CLIverResponse(message=CLIverMessage(role="assistant", content="done"), usage=None, media=None)

    async def stream(self, request):  # pragma: no cover — chat test only
        if False:
            yield None


class _ActivatingStreamProvider(Provider):
    protocol = "openai"
    brand = "mock-activate-stream"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used by the streaming loop")

    async def stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield CLIverMessageChunk(
                native={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "ActivateSkill", "arguments": '{"name": "test-greeting"}'},
                        }
                    ],
                }
            )
        else:
            yield CLIverMessageChunk(content="done")
            yield CLIverMessageChunk(native={"role": "assistant", "content": "done"})


async def test_chat_narrows_tools_after_activation():
    provider = _ActivatingProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[_make_tool("Read"), _make_tool("WebSearch")],
        skill_manager=_manager(),
    )

    response = await agent.chat("greet me")

    assert response.message.content == "done"
    assert [tool.name for tool in provider.requests[0].tools] == [
        "Read",
        "WebSearch",
        "FindSkills",
        "ActivateSkill",
    ]
    assert [tool.name for tool in provider.requests[1].tools] == ["WebSearch", "ReadSkillFile"]
    tool_message = next(m for m in provider.requests[1].messages if m.role == "tool")
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.content.startswith("# Active skill: test-greeting")
    assert f"Skill location: {FIXTURES / 'test-greeting'}" in tool_message.content
    assert "references/phrases.md" in tool_message.content


async def test_stream_narrows_tools_after_activation():
    provider = _ActivatingStreamProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[_make_tool("Read"), _make_tool("WebSearch")],
        skill_manager=_manager(),
    )

    chunks = [chunk async for chunk in agent.stream("greet me")]

    assert "".join(chunk.content or "" for chunk in chunks) == "done"
    assert [tool.name for tool in provider.requests[1].tools] == ["WebSearch", "ReadSkillFile"]


async def test_activation_without_bundled_files_does_not_advertise_read_skill_file():
    provider = _ActivatingProvider(skill_name="test-code-review")
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[_make_tool("Read")],
        skill_manager=_manager(),
    )

    await agent.chat("review this")

    assert [tool.name for tool in provider.requests[1].tools] == [
        "Read",
        "FindSkills",
        "ActivateSkill",
    ]


async def test_read_skill_file_reads_a_bundled_file_through_the_agent():
    provider = _ReadSkillFileProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[_make_tool("Read"), _make_tool("WebSearch")],
        skill_manager=_manager(),
    )

    response = await agent.chat("greet me")

    assert response.message.content == "done"
    assert [tool.name for tool in provider.requests[1].tools] == ["WebSearch", "ReadSkillFile"]
    assert [tool.name for tool in provider.requests[2].tools] == ["WebSearch", "ReadSkillFile"]
    tool_message = next(m for m in provider.requests[2].messages if m.role == "tool" and m.tool_call_id == "call_2")
    assert "Aloha means hello" in tool_message.content


async def test_read_skill_file_name_is_reserved_when_dispatch_is_on():
    agent = _agent(builtin_tools=[_make_tool("ReadSkillFile")], skill_manager=_manager())

    with pytest.raises(ValueError, match="reserved for skill dispatch"):
        await agent.chat("hi")
