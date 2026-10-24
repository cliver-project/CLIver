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
from cliver_core.tools.execute_code import execute_code
from cliver_core.tools.grep_search import grep_search
from cliver_core.tools.list_directory import list_directory
from cliver_core.tools.read_file import read_file
from cliver_core.tools.run_shell_command import run_shell_command
from cliver_core.tools.write_file import write_file

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
    from cliver_core.tools import discover_builtin_tools

    by_name = {tool.name: tool for tool in discover_builtin_tools()}
    assert by_name["Read"].safety == "safe"
    assert by_name["LS"].safety == "safe"
    assert by_name["Grep"].safety == "safe"
    assert by_name["TodoRead"].safety == "safe"  # agent-internal scratchpad
    assert by_name["TodoWrite"].safety == "safe"
    assert by_name["Write"].safety == "mutating"
    assert by_name["Bash"].safety == "mutating"
    assert by_name["Exec"].safety == "mutating"
    assert by_name["WebFetch"].safety == "mutating"


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
    """The effective policy is ambient during tool execution (its only
    consumer) and nowhere else."""
    agent = _policy_agent(execution_policy=ExecutionPolicy())
    seen: list[ExecutionPolicy | None] = []
    outside: list[ExecutionPolicy | None] = []

    @tool(name="probe", description="Records the ambient policy.")
    def probe() -> list[dict]:
        seen.append(current_policy())
        return [{"text": "ok"}]

    kwarg = ExecutionPolicy(env=EnvSpec(mode="allowlist", names={"PATH"}))

    class _ToolThenTextProvider(_NoToolProvider):
        protocol = "openai"

        async def chat(self, request: CLIverRequest) -> CLIverResponse:
            self.calls.append(request)
            if len(self.calls) % 2 == 1:
                return CLIverResponse(
                    message=CLIverMessage(
                        role="assistant", content=None, tool_calls=[ToolCall(id="c", name="probe", args={})]
                    ),
                    usage=None,
                    media=None,
                )
            return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"), usage=None, media=None)

    tool_agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=_ToolThenTextProvider())),
        builtin_tools=[probe],
        execution_policy=ExecutionPolicy(),
    )

    await tool_agent.chat("hi")  # constructor default applies inside tools
    await tool_agent.chat("hi", execution_policy=kwarg)  # call-site wins
    outside.append(current_policy())  # not leaked after chat returns

    assert seen[0] is tool_agent.execution_policy
    assert seen[1] is kwarg
    assert outside == [None]
    assert agent.execution_policy is not None  # fixture still constructs with a default


# ── Bash/Exec honor the ambient execution policy ──────────────


async def test_bash_env_allowlist_scrubs(monkeypatch):
    monkeypatch.setenv("CLIVER_TEST_SECRET", "hunter2")
    policy = ExecutionPolicy(env=EnvSpec(mode="allowlist", names=set(SAFE_ENV_VARS)))

    with policy_scope(policy):
        result = run_shell_command.execute(command="env")

    text = result[0]["text"]
    assert "CLIVER_TEST_SECRET" not in text
    assert "PATH" in text  # the allowlist keeps what it names


async def test_bash_env_inherited_without_policy(monkeypatch):
    monkeypatch.setenv("CLIVER_TEST_SECRET", "hunter2")

    result = run_shell_command.execute(command="env")

    assert "CLIVER_TEST_SECRET" in result[0]["text"]


async def test_bash_cwd_pinned_and_directory_overridden(tmp_path, caplog):
    other = tmp_path / "other"
    other.mkdir()
    policy = ExecutionPolicy(cwd=tmp_path)

    with caplog.at_level("INFO", logger="cliver_core.tools.run_shell_command"):
        with policy_scope(policy):
            result = run_shell_command.execute(command="pwd", directory=str(other))

    assert result[0]["text"].strip() == str(tmp_path)
    override_logs = [r for r in caplog.records if "pins cwd" in r.getMessage()]
    assert len(override_logs) == 1
    assert str(other) in override_logs[0].getMessage()


async def test_bash_directory_honored_when_no_cwd_pin(tmp_path):
    target = tmp_path / "sub"
    target.mkdir()
    policy = ExecutionPolicy(cwd=None)

    with policy_scope(policy):
        result = run_shell_command.execute(command="pwd", directory=str(target))

    assert result[0]["text"].strip() == str(target)


async def test_exec_env_and_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIVER_TEST_SECRET", "hunter2")
    policy = ExecutionPolicy(cwd=tmp_path, env=EnvSpec(mode="allowlist", names=set(SAFE_ENV_VARS)))
    code = "import os; print(os.getcwd()); print(os.environ.get('CLIVER_TEST_SECRET', 'absent'))"

    with policy_scope(policy):
        result = execute_code.execute(code=code)

    lines = result[0]["text"].strip().splitlines()
    assert lines[0] == str(tmp_path)
    assert lines[1] == "absent"


# ── File tools enforce execution-policy scopes ────────────────


def test_write_outside_write_paths_rejected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    policy = ExecutionPolicy(write_paths=[data])

    with policy_scope(policy):
        result = write_file.execute(file_path=str(tmp_path / "outside.txt"), content="x")

    assert "outside the allowed write paths" in result[0]["error"]
    assert not (tmp_path / "outside.txt").exists()


def test_write_inside_write_paths(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    policy = ExecutionPolicy(write_paths=[data])

    with policy_scope(policy):
        result = write_file.execute(file_path=str(data / "allowed.txt"), content="hello")

    assert "error" not in result[0]
    assert (data / "allowed.txt").read_text() == "hello"


def test_read_outside_read_paths_rejected(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    target = tmp_path / "plain.txt"
    target.write_text("s3cret")
    policy = ExecutionPolicy(read_paths=[inside])

    with policy_scope(policy):
        result = read_file.execute(file_path=str(target))

    assert "outside the allowed read paths" in result[0]["error"]


def test_read_unrestricted_when_none(tmp_path):
    target = tmp_path / "anywhere.txt"
    target.write_text("visible")
    policy = ExecutionPolicy()  # read_paths None → unrestricted

    with policy_scope(policy):
        result = read_file.execute(file_path=str(target))

    assert "visible" in result[0]["text"]


def test_list_directory_and_grep_respect_read_paths(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / "note.md").write_text("needle here")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("needle there")
    policy = ExecutionPolicy(read_paths=[inside])

    with policy_scope(policy):
        ls_out = list_directory.execute(path=str(outside))
        grep_out = grep_search.execute(pattern="needle", path=str(outside))
        ls_in = list_directory.execute(path=str(inside))
        grep_in = grep_search.execute(pattern="needle", path=str(inside))

    assert "outside the allowed read paths" in ls_out[0]["error"]
    assert "outside the allowed read paths" in grep_out[0]["error"]
    assert "note.md" in ls_in[0]["text"]
    assert "needle here" in grep_in[0]["text"]


def test_symlink_escape_rejected_end_to_end(tmp_path):
    jail = tmp_path / "jail"
    jail.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("stolen")
    (jail / "link.txt").symlink_to(victim)
    policy = ExecutionPolicy(read_paths=[jail])

    with policy_scope(policy):
        result = read_file.execute(file_path=str(jail / "link.txt"))

    assert "outside the allowed read paths" in result[0]["error"]


def test_relative_paths_resolve_under_pinned_cwd(tmp_path, monkeypatch):
    """The scope check and the file access must agree on where a relative
    path lives: under the pinned cwd, not the process cwd."""
    jail = tmp_path / "jail"
    jail.mkdir()
    (jail / "note.md").write_text("needle here")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # process cwd outside the scope
    policy = ExecutionPolicy(cwd=jail, read_paths=[jail], write_paths=[jail])

    with policy_scope(policy):
        grep = grep_search.execute(pattern="needle")  # default path="."
        read = read_file.execute(file_path="note.md")
        write_file.execute(file_path="out/x.txt", content="data")

    assert "needle here" in grep[0]["text"]
    assert "needle here" in read[0]["text"]
    assert (jail / "out" / "x.txt").read_text() == "data"
    assert not (elsewhere / "out").exists()  # nothing escaped the pin


async def test_stream_policy_not_leaked_when_abandoned():
    """Breaking out of a stream must not leave the policy ambient in the
    consumer's task (the scope never spans yields)."""
    from cliver_core.messages import CLIverMessageChunk

    agent = _policy_agent(execution_policy=ExecutionPolicy())

    async def fake_stream_loop(messages, tools, options, max_iterations, execution_policy=None, skill_state=None):
        yield CLIverMessageChunk(content="partial")
        yield CLIverMessageChunk(content="more")

    agent._run_stream_loop = fake_stream_loop

    async for _chunk in agent.stream("hi", execution_policy=agent.execution_policy):
        break  # abandon without aclose

    assert current_policy() is None
