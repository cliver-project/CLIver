"""AgentCore must start the MCP client before gathering tools.

``MCPClient.get_tools()`` is empty until ``start()`` has discovered tools, so
gathering first means the first turn of a session advertises zero MCP tools.
"""

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.tool import CLIverTool

_ECHO_TOOL = CLIverTool(
    name="mcp__demo__echo",
    description="Echo a message",
    parameters={"type": "object", "properties": {}},
    execute=lambda **_kwargs: [{"text": "ok"}],
)


class _OrderRecordingMCPClient:
    """Stub client: tools exist only after ``start()``; records call order."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = False

    async def start(self) -> None:
        self.calls.append("start")
        self.started = True

    async def get_tools(self, servers: list[str] | None = None) -> list[CLIverTool]:
        self.calls.append("get_tools")
        return [_ECHO_TOOL] if self.started else []


def _agent(client: _OrderRecordingMCPClient) -> AgentCore:
    return AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model")), mcp_client=client)


async def test_chat_starts_mcp_client_before_gathering_tools():
    client = _OrderRecordingMCPClient()
    agent = _agent(client)
    captured: dict = {}

    async def fake_run_loop(messages, all_tools, opts, max_iterations):
        captured["tools"] = all_tools
        return None

    agent._run_loop = fake_run_loop  # type: ignore[method-assign]
    await agent.chat("hello")

    assert client.calls[:2] == ["start", "get_tools"]
    assert [tool.name for tool in captured["tools"]] == ["mcp__demo__echo"]


async def test_stream_starts_mcp_client_before_gathering_tools():
    client = _OrderRecordingMCPClient()
    agent = _agent(client)
    captured: dict = {}

    async def fake_run_stream(messages, all_tools, opts, max_iterations):
        captured["tools"] = all_tools
        return
        yield  # pragma: no cover — makes this an async generator

    agent._run_stream_loop = fake_run_stream  # type: ignore[method-assign]
    assert [chunk async for chunk in agent.stream("hello")] == []

    assert client.calls[:2] == ["start", "get_tools"]
    assert [tool.name for tool in captured["tools"]] == ["mcp__demo__echo"]


async def test_repeated_turns_keep_mcp_tools_advertised():
    client = _OrderRecordingMCPClient()
    agent = _agent(client)
    seen: list[list[str]] = []

    async def fake_run_loop(messages, all_tools, opts, max_iterations):
        seen.append([tool.name for tool in all_tools])
        return None

    agent._run_loop = fake_run_loop  # type: ignore[method-assign]
    await agent.chat("first")
    await agent.chat("second")

    assert seen == [["mcp__demo__echo"], ["mcp__demo__echo"]]
    assert client.calls == ["start", "get_tools", "start", "get_tools"]
