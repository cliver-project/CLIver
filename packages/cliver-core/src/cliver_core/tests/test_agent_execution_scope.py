"""Per-call execution scope and per-run builtin tool state.

The advertised tool list is an execution boundary, not just a suggestion: a
tool the model calls but was not advertised — filtered out, disallowed by a
skill's ``allowed-tools``, or belonging to an MCP server not selected for the
call — must never run and must not trigger a permission prompt.

Builtin Todo state belongs to one ``chat()`` / ``stream()`` run.  A later or
concurrent run must never observe another run's list.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.events import ToolEventType
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall
from cliver_core.policy import Decision, ToolCallContext
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.skill import SkillManager
from cliver_core.tool import CLIverTool
from cliver_core.tools import discover_builtin_tools


class _ScriptedProvider(Provider):
    protocol = "openai"
    brand = "mock-execution-scope"

    def __init__(self, responses: list[CLIverResponse]) -> None:
        super().__init__(api_key="", base_url="")
        self.responses = list(responses)
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


class _BarrierProvider(_ScriptedProvider):
    """Waits for every run's second model call before answering it.

    Both runs finish their TodoWrite tool batch before either run's TodoRead
    turn is answered, so a shared module-level todo list would be observed by
    both reads.
    """

    def __init__(self, responses: list[CLIverResponse], barrier: "_Barrier") -> None:
        super().__init__(responses)
        self.barrier = barrier

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        if len(self.requests) == 2:
            await self.barrier.wait()
        return self.responses.pop(0)


class _Barrier:
    """Async barrier (``asyncio.Barrier`` needs Python 3.11)."""

    def __init__(self, parties: int) -> None:
        self.parties = parties
        self.arrived = 0
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            self.arrived += 1
            if self.arrived >= self.parties:
                self.event.set()
        await self.event.wait()


def _final(text: str = "done") -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text))


def _call(call_id: str, name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args or {})


def _tool_turn(*calls: ToolCall) -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=None, tool_calls=list(calls)))


def _tool(name: str, execute, *, safety: str = "safe", origin: str = "local") -> CLIverTool:
    return CLIverTool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
        safety=safety,
        origin=origin,
    )


def _agent_with(
    provider: Provider,
    tools: list[CLIverTool],
    **options,
) -> AgentCore:
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider))
    return AgentCore(bindings, builtin_tools=list(tools), **options)


def _agent(
    responses: list[CLIverResponse],
    tools: list[CLIverTool],
    **options,
) -> tuple[AgentCore, _ScriptedProvider]:
    provider = _ScriptedProvider(responses)
    return _agent_with(provider, tools, **options), provider


def _last_tool_text(request: CLIverRequest) -> str:
    tool_messages = [message for message in request.messages if message.role == "tool"]
    assert tool_messages
    return tool_messages[-1].content or ""


# ── Advertised set is the execution set ───────────────────────


@pytest.mark.asyncio
async def test_filtered_tool_never_runs_and_never_prompts() -> None:
    prompts: list[str] = []
    blocked_runs: list[int] = []

    async def permission(ctx: ToolCallContext) -> Decision:
        prompts.append(ctx.tool_name)
        return Decision.allow()

    blocked = _tool(
        "blocked",
        lambda **_kwargs: blocked_runs.append(1) or [{"text": "blocked ran"}],
        safety="mutating",
    )
    allowed = _tool("allowed", lambda **_kwargs: [{"text": "allowed ran"}])
    agent, provider = _agent(
        [_tool_turn(_call("c1", "blocked")), _final()],
        [blocked, allowed],
        permission_callback=permission,
    )
    events = []

    async def on_event(event) -> None:
        events.append(event)

    response = await agent.chat(
        "go",
        tool_filter=lambda tool: tool.name == "allowed",
        on_event=on_event,
    )

    assert response.message.content == "done"
    assert blocked_runs == []
    assert prompts == []
    assert "not available for this call" in _last_tool_text(provider.requests[-1])
    blocked_events = [event for event in events if event.tool_name == "blocked"]
    assert [event.event for event in blocked_events] == [ToolEventType.START, ToolEventType.ERROR]
    assert "not available for this call" in (blocked_events[-1].error or "")


@pytest.mark.asyncio
async def test_unadvertised_tool_is_rejected_in_parallel_mode() -> None:
    prompts: list[str] = []
    runs: list[str] = []

    async def permission(ctx: ToolCallContext) -> Decision:
        prompts.append(ctx.tool_name)
        return Decision.allow()

    allowed = _tool("allowed", lambda **_kwargs: runs.append("allowed") or [{"text": "ok"}])
    blocked = _tool(
        "blocked",
        lambda **_kwargs: runs.append("blocked") or [{"text": "blocked ran"}],
        safety="mutating",
    )
    agent, provider = _agent(
        [_tool_turn(_call("c1", "allowed"), _call("c2", "blocked")), _final()],
        [allowed, blocked],
        permission_callback=permission,
    )

    await agent.chat(
        "go",
        tool_filter=lambda tool: tool.name == "allowed",
        parallel_tool_calls=True,
    )

    assert runs == ["allowed"]
    assert prompts == []
    tool_messages = [message for message in provider.requests[-1].messages if message.role == "tool"]
    assert "ok" in (tool_messages[0].content or "")
    assert "not available for this call" in (tool_messages[1].content or "")


@pytest.mark.asyncio
async def test_skill_allowed_tools_is_an_execution_boundary(tmp_path) -> None:
    skill_dir = tmp_path / "scope"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: scope\ndescription: Only allowed tools.\nallowed-tools: allowed\n---\n\n# Scope\n",
        encoding="utf-8",
    )
    runs: list[str] = []
    blocked = _tool(
        "blocked",
        lambda **_kwargs: runs.append("blocked") or [{"text": "blocked ran"}],
    )
    allowed = _tool("allowed", lambda **_kwargs: runs.append("allowed") or [{"text": "ok"}])
    agent, provider = _agent(
        [_tool_turn(_call("c1", "blocked")), _final()],
        [blocked, allowed],
        skill_manager=SkillManager(sources=[tmp_path]),
    )

    await agent.run_skill("scope", "go")

    assert runs == []
    assert "not available for this call" in _last_tool_text(provider.requests[-1])


# ── MCP per-call server scope ─────────────────────────────────


class _StubMCPClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = False
        self.tools_by_server = {
            "a": _tool("mcp__a__echo", lambda **_kwargs: [{"text": "a"}], origin="mcp"),
            "b": _tool("mcp__b__secret", lambda **_kwargs: [{"text": "b"}], origin="mcp"),
        }

    async def start(self) -> None:
        self.started = True

    async def get_tools(self, servers: list[str] | None = None) -> list[CLIverTool]:
        if servers is None:
            return list(self.tools_by_server.values())
        return [self.tools_by_server[name] for name in servers if name in self.tools_by_server]

    async def call_tool(self, name: str, args: dict) -> list[dict]:
        self.calls.append(name)
        return [{"text": f"called {name}"}]


@pytest.mark.asyncio
async def test_unselected_mcp_server_tools_cannot_run() -> None:
    client = _StubMCPClient()
    agent, provider = _agent(
        [
            _tool_turn(_call("c1", "mcp__a__echo"), _call("c2", "mcp__b__secret")),
            _final(),
        ],
        [],
        mcp_client=client,
    )

    await agent.chat("go", mcp_servers=["a"])

    assert client.calls == ["mcp__a__echo"]
    tool_messages = [message for message in provider.requests[-1].messages if message.role == "tool"]
    assert "called mcp__a__echo" in (tool_messages[0].content or "")
    assert "not available for this call" in (tool_messages[1].content or "")


@pytest.mark.asyncio
async def test_empty_mcp_server_filter_advertises_and_runs_nothing() -> None:
    client = _StubMCPClient()
    agent, provider = _agent(
        [_tool_turn(_call("c1", "mcp__a__echo")), _final()],
        [],
        mcp_client=client,
    )

    await agent.chat("go", mcp_servers=[])

    assert client.calls == []
    assert "not available for this call" in _last_tool_text(provider.requests[-1])


@pytest.mark.asyncio
async def test_local_tool_with_mcp_prefix_is_not_routed_to_mcp() -> None:
    """Routing follows tool identity, not the name's shape."""
    client = _StubMCPClient()
    runs: list[str] = []
    local = _tool(
        "mcp__local__echo",
        lambda **_kwargs: runs.append("local") or [{"text": "local result"}],
        origin="local",
    )
    agent, provider = _agent(
        [_tool_turn(_call("c1", "mcp__local__echo")), _final()],
        [local],
        mcp_client=client,
    )

    await agent.chat("go")

    assert runs == ["local"]
    assert client.calls == []
    assert "local result" in _last_tool_text(provider.requests[-1])


# ── Per-run Todo state ────────────────────────────────────────


def _todo_tools() -> list[CLIverTool]:
    return [tool for tool in discover_builtin_tools() if tool.name in {"TodoRead", "TodoWrite"}]


def _todo_write(call_id: str, item: str) -> ToolCall:
    return _call(
        call_id,
        "TodoWrite",
        {"todos": [{"id": item, "content": item, "status": "pending"}]},
    )


def _todo_read(call_id: str) -> ToolCall:
    return _call(call_id, "TodoRead")


@pytest.mark.asyncio
async def test_todo_state_does_not_leak_between_runs() -> None:
    agent, provider = _agent(
        [
            # Run A: write alpha, then read it back.
            _tool_turn(_todo_write("a1", "alpha")),
            _tool_turn(_todo_read("a2")),
            _final("A done"),
            # Run B: replace the list.
            _tool_turn(_todo_write("b1", "beta")),
            _final("B done"),
            # Run C: a fresh run starts with no plan.
            _tool_turn(_todo_read("c1")),
            _final("C done"),
        ],
        _todo_tools(),
    )

    await agent.chat("A")
    await agent.chat("B")
    await agent.chat("C")

    # The read result is carried into the *next* model request.
    a_read = _last_tool_text(provider.requests[2])
    assert "alpha" in a_read
    assert "beta" not in a_read
    assert _last_tool_text(provider.requests[6]) == "No active plan."


@pytest.mark.asyncio
async def test_concurrent_runs_keep_separate_todo_lists() -> None:
    tools = _todo_tools()
    barrier = _Barrier(2)
    provider_a = _BarrierProvider(
        [
            _tool_turn(_todo_write("a1", "alpha")),
            _tool_turn(_todo_read("a2")),
            _final("A done"),
        ],
        barrier,
    )
    provider_b = _BarrierProvider(
        [
            _tool_turn(_todo_write("b1", "beta")),
            _tool_turn(_todo_read("b2")),
            _final("B done"),
        ],
        barrier,
    )
    agent_a = _agent_with(provider_a, tools)
    agent_b = _agent_with(provider_b, tools)

    await asyncio.gather(agent_a.chat("A"), agent_b.chat("B"))

    a_read = _last_tool_text(provider_a.requests[2])
    b_read = _last_tool_text(provider_b.requests[2])
    assert "alpha" in a_read and "beta" not in a_read
    assert "beta" in b_read and "alpha" not in b_read


class _ScriptedStreamingProvider(Provider):
    """Streams scripted native turns; records each request."""

    protocol = "openai"
    brand = "mock-execution-scope-stream"

    def __init__(self, natives: list[dict]) -> None:
        super().__init__(api_key="", base_url="")
        self.natives = list(natives)
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used here")

    async def stream(self, request):
        self.requests.append(request)
        native = self.natives.pop(0)
        if native.get("tool_calls"):
            yield CLIverMessageChunk(native=native)
            return
        yield CLIverMessageChunk(content=native.get("content") or "")
        yield CLIverMessageChunk(native=native)


def _native_call(call: ToolCall) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.args)},
            }
        ],
    }


@pytest.mark.asyncio
async def test_stream_run_keeps_its_own_todo_state() -> None:
    provider = _ScriptedStreamingProvider(
        [
            _native_call(_todo_write("a1", "alpha")),
            _native_call(_todo_read("a2")),
            {"role": "assistant", "content": "A done"},
            _native_call(_todo_read("c1")),
            {"role": "assistant", "content": "C done"},
        ]
    )
    agent = _agent_with(provider, _todo_tools())

    async for _chunk in agent.stream("A"):
        pass
    async for _chunk in agent.stream("C"):
        pass

    a_read = _last_tool_text(provider.requests[2])
    assert "alpha" in a_read
    assert _last_tool_text(provider.requests[4]) == "No active plan."
