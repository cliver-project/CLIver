"""Tool policy: decisions, env specs, execution scopes, and the ambient policy context."""

import asyncio
from pathlib import Path

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.events import ToolEventType
from cliver_core.messages import CLIverMessage, ToolCall
from cliver_core.policy import (
    SAFE_ENV_VARS,
    Decision,
    DecisionAction,
    EnvSpec,
    ExecutionPolicy,
    ToolCallContext,
    ToolCategory,
    current_policy,
    policy_scope,
    scope_violation,
)
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.tool import tool
from cliver_core.tools import read_file, run_shell_command

# ── Decisions and categories ──────────────────────────────────


def test_decision_allow_and_deny():
    allow = Decision.allow()
    deny = Decision.deny("not on this server")

    assert allow.action is DecisionAction.ALLOW and allow.reason is None
    assert deny.action is DecisionAction.DENY and deny.reason == "not on this server"


def test_tool_call_context_carries_server_and_category():
    ctx = ToolCallContext(
        tool_name="mcp__everything__get_weather",
        args={"city": "Beijing"},
        server="everything",
        category=ToolCategory.ASK,
    )

    assert ctx.server == "everything"
    assert ctx.category is ToolCategory.ASK


# ── EnvSpec ───────────────────────────────────────────────────


def test_env_spec_inherit():
    source = {"PATH": "/bin", "SECRET": "x"}

    assert EnvSpec().build(source) == source
    assert EnvSpec().build(source) is not source  # a copy, never the caller's dict


def test_env_spec_allowlist_only_lists_names():
    built = EnvSpec(mode="allowlist", names={"PATH", "HOME", "MISSING"}).build(
        {"PATH": "/bin", "HOME": "/u", "SECRET": "x"}
    )

    assert built == {"PATH": "/bin", "HOME": "/u"}  # SECRET dropped, MISSING absent


def test_env_spec_denylist_drops_names():
    built = EnvSpec(mode="denylist", names={"SECRET"}).build({"PATH": "/bin", "SECRET": "x"})

    assert built == {"PATH": "/bin"}


def test_safe_env_vars_starter_set():
    assert {"PATH", "HOME", "TMPDIR", "LANG"} <= SAFE_ENV_VARS


# ── ExecutionPolicy defaults ──────────────────────────────────


def test_execution_policy_defaults():
    policy = ExecutionPolicy()

    assert policy.env.mode == "inherit" and policy.env.names == set()
    assert policy.cwd is None
    assert policy.read_paths is None
    assert policy.write_paths is None


# ── Ambient policy context ────────────────────────────────────


def test_policy_scope_sets_and_resets():
    outer = ExecutionPolicy()
    inner = ExecutionPolicy(cwd=Path("/tmp/inner"))

    assert current_policy() is None
    with policy_scope(outer):
        assert current_policy() is outer
        with policy_scope(inner):
            assert current_policy() is inner
        assert current_policy() is outer
    assert current_policy() is None


def test_policy_scope_none_is_noop():
    with policy_scope(None):
        assert current_policy() is None


async def test_policy_scope_isolated_across_concurrent_tasks():
    observed: list[bool] = []

    async def hold(policy: ExecutionPolicy) -> None:
        with policy_scope(policy):
            await asyncio.sleep(0.01)
            observed.append(current_policy() is policy)

    await asyncio.gather(hold(ExecutionPolicy()), hold(ExecutionPolicy(cwd=Path("/tmp/other"))))

    assert observed == [True, True]


# ── Path scopes ───────────────────────────────────────────────


def test_scope_violation_read_vs_write(tmp_path):
    root = tmp_path / "scope"
    root.mkdir()
    (root / "out").mkdir()
    policy = ExecutionPolicy(read_paths=[root], write_paths=[root / "out"])

    with policy_scope(policy):
        assert scope_violation(root / "doc.md", write=False) is None
        # Inside the read scope but not the write scope.
        assert scope_violation(root / "doc.md", write=True) is not None
        assert scope_violation(root / "out" / "f.txt", write=True) is None
        assert scope_violation(tmp_path / "elsewhere.md", write=False) is not None


def test_scope_violation_symlink_escape(tmp_path):
    scope = tmp_path / "scope"
    scope.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = scope / "leak"
    link.symlink_to(outside)

    with policy_scope(ExecutionPolicy(read_paths=[scope])):
        assert scope_violation(link, write=False) is not None


def test_scope_violation_resolves_against_policy_cwd(tmp_path):
    with policy_scope(ExecutionPolicy(read_paths=[tmp_path], cwd=tmp_path)):
        assert scope_violation("doc.md", write=False) is None
    with policy_scope(ExecutionPolicy(read_paths=[tmp_path], cwd=tmp_path.parent)):
        # "doc.md" resolves under cwd — outside the read scope this time.
        assert scope_violation("doc.md", write=False) is not None


def test_scope_violation_unrestricted_when_none(tmp_path):
    with policy_scope(ExecutionPolicy()):  # scopes None
        assert scope_violation(tmp_path / "anywhere.md", write=True) is None
    assert scope_violation(tmp_path / "anywhere.md", write=False) is None  # no policy


# ── Builtin safety flags ──────────────────────────────────────


def test_builtin_safety_flags():
    from cliver_core.tools import (
        execute_code,
        grep_search,
        list_directory,
        read_file,
        run_shell_command,
        todo_read,
        todo_write,
        web_fetch,
        write_file,
    )

    assert read_file.safety == "safe"
    assert list_directory.safety == "safe"
    assert grep_search.safety == "safe"
    assert todo_read.safety == "safe"  # agent-internal scratchpad
    assert todo_write.safety == "safe"
    assert write_file.safety == "mutating"
    assert run_shell_command.safety == "mutating"
    assert execute_code.safety == "mutating"
    assert web_fetch.safety == "mutating"


# ── AgentCore seam: permission gate and denial semantics ──────


class _NoToolProvider(Provider):
    """Answers with plain text — the permission gate is exercised directly."""

    protocol = "mock"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"), usage=None, media=None)

    async def stream(self, request):
        if False:  # pragma: no cover — not exercised here
            yield None


@tool(name="poke", description="Mutating test tool.")
def poke(value: str) -> list[dict]:
    return [{"text": f"poked {value}"}]


@tool(name="peek", description="Safe test tool.", safety="safe")
def peek(value: str) -> list[dict]:
    return [{"text": f"peeked {value}"}]


def _policy_agent(**kwargs) -> AgentCore:
    return AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=_NoToolProvider())),
        builtin_tools=[poke, peek],
        **kwargs,
    )


async def test_deny_returns_denied_result():
    seen: list[ToolCallContext] = []

    async def deny(ctx: ToolCallContext) -> Decision:
        seen.append(ctx)
        return Decision.deny("not allowed here")

    agent = _policy_agent(permission_callback=deny)

    result = await agent._execute_tool(ToolCall(id="c1", name="poke", args={"value": "x"}), poke)

    assert result == [{"error": "Permission denied: not allowed here", "denied": True}]
    assert [c.tool_name for c in seen] == ["poke"]
    assert seen[0].category is ToolCategory.ASK
    assert seen[0].server is None
    assert seen[0].args == {"value": "x"}


async def test_safe_tool_skips_callback():
    async def deny(ctx: ToolCallContext) -> Decision:  # would deny everything
        return Decision.deny("no")

    agent = _policy_agent(permission_callback=deny)

    result = await agent._execute_tool(ToolCall(id="c1", name="peek", args={"value": "x"}), peek)

    assert result == [{"text": "peeked x"}]


async def test_ask_tool_allowed_runs():
    async def allow(ctx: ToolCallContext) -> Decision:
        return Decision.allow()

    agent = _policy_agent(permission_callback=allow)

    result = await agent._execute_tool(ToolCall(id="c1", name="poke", args={"value": "x"}), poke)

    assert result == [{"text": "poked x"}]


async def test_no_callback_runs_unchanged():
    agent = _policy_agent()

    result = await agent._execute_tool(ToolCall(id="c1", name="poke", args={"value": "x"}), poke)

    assert result == [{"text": "poked x"}]


async def test_safety_overrides_strict_and_loose():
    consults: list[str] = []

    async def deny(ctx: ToolCallContext) -> Decision:
        consults.append(ctx.tool_name)
        return Decision.deny("no")

    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=_NoToolProvider())),
        builtin_tools=[run_shell_command, read_file],
        permission_callback=deny,
        safety_overrides={"Bash": ToolCategory.SAFE, "Read": ToolCategory.ASK},
    )

    # Bash: default mutating, overridden to SAFE — runs, callback never asked.
    bash = await agent._execute_tool(ToolCall(id="c1", name="Bash", args={"command": "echo ok"}), run_shell_command)
    assert "ok" in bash[0]["text"]
    assert consults == []

    # Read: default safe, overridden to ASK — consults and is denied.
    read = await agent._execute_tool(ToolCall(id="c2", name="Read", args={"file_path": "/etc/hostname"}), read_file)
    assert read == [{"error": "Permission denied: no", "denied": True}]
    assert consults == ["Read"]


async def test_callback_exception_fails_closed():
    async def boom(ctx: ToolCallContext) -> Decision:
        raise RuntimeError("callback crashed")

    agent = _policy_agent(permission_callback=boom)

    result = await agent._execute_tool(ToolCall(id="c1", name="poke", args={"value": "x"}), poke)

    assert result == [{"error": "Permission denied: permission callback failed: callback crashed", "denied": True}]


class _AlwaysToolProvider(_NoToolProvider):
    """Every turn requests ``poke`` — the loop only stops via the denial cap."""

    protocol = "openai"
    brand = "mock-deny"

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        return CLIverResponse(
            message=CLIverMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id=f"call_{len(self.calls)}", name="poke", args={"value": "x"})],
            ),
            usage=None,
            media=None,
        )


async def test_denial_cap_stops_loop():
    async def deny(ctx: ToolCallContext) -> Decision:
        return Decision.deny("always no")

    events = []

    async def record(event) -> None:
        events.append(event)

    provider = _AlwaysToolProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[poke],
        permission_callback=deny,
        on_event=record,
    )

    response = await agent.chat("hi")

    assert "failed repeatedly" in response.message.content
    assert len(provider.calls) == 5  # five turns, then the loop stopped
    ends = [e for e in events if e.event is ToolEventType.END]
    assert len(ends) == 5  # five denials — each through the END-event path
    assert "Permission denied: always no" in ends[-1].result


async def test_chat_execution_policy_flows():
    agent = _policy_agent(execution_policy=ExecutionPolicy())
    seen: list[ExecutionPolicy | None] = []

    async def fake_loop(messages, tools, options, max_iterations):
        seen.append(current_policy())
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"), usage=None, media=None)

    agent._run_loop = fake_loop

    await agent.chat("hi")  # constructor default applies
    kwarg = ExecutionPolicy(env=EnvSpec(mode="allowlist", names={"PATH"}))
    await agent.chat("hi", execution_policy=kwarg)  # call-site wins

    assert seen == [agent.execution_policy, kwarg]
