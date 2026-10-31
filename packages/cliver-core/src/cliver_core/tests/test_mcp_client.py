"""MCPClient routing and lifecycle — the adapter factory is stubbed (no SDK).

``cliver_core.mcp._adapters`` is replaced with a recording stub so these
tests pin the client-level contract: ``mcp__<server>__<tool>`` parsing,
lifecycle, and the prompt/resource passthrough shapes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cliver_core.mcp import MCPClient
from cliver_core.tool import CLIverTool


class StubAdapter:
    """Records lifecycle calls; exposes one tool named ``mcp__<server>__echo``."""

    instances: list["StubAdapter"] = []

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.started = 0
        self.closed = 0
        self.calls: list[tuple[str, dict]] = []
        StubAdapter.instances.append(self)

    async def start(self) -> None:
        if self.config.get("fail_start"):
            raise RuntimeError("boom")
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def call_tool(self, tool_name: str, args: dict) -> list[dict]:
        self.calls.append((tool_name, args))
        return [{"text": f"{self.name}:{tool_name}"}]

    async def list_prompts(self) -> list[dict]:
        return [{"prompts": [{"name": "demo", "description": "A demo prompt", "arguments": []}]}]

    async def get_prompt(self, name: str, arguments: dict | None = None) -> list[dict]:
        return [{"messages": [{"text": f"{name}:{arguments}"}]}]

    async def list_resources(self) -> list[dict]:
        return [{"resources": [{"uri": "test://demo", "description": "A demo resource"}]}]

    async def read_resource(self, uri: str) -> list[dict]:
        return [{"contents": [{"uri": uri, "text": "demo body"}]}]

    @property
    def tools(self) -> list[CLIverTool]:
        if not self.started:
            return []
        return [
            CLIverTool(
                name=f"mcp__{self.name}__echo",
                description="Echo",
                parameters={"type": "object", "properties": {}},
                execute=lambda **kwargs: [],
            )
        ]


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setattr("cliver_core.mcp._adapters", lambda: (object, StubAdapter))
    StubAdapter.instances = []
    return StubAdapter


async def test_call_tool_routes_on_mcp_prefix(stub):
    client = MCPClient({"srv": {}})

    result = await client.call_tool("mcp__srv__echo", {"message": "hi"})

    assert result == [{"text": "srv:echo"}]
    assert stub.instances[0].calls == [("echo", {"message": "hi"})]  # bare tool name


async def test_call_tool_rejects_malformed_names(stub):
    client = MCPClient({"srv": {}})

    for bad in ("echo", "mcp__srv", "mcp__srv__", "mcp__"):
        (result,) = await client.call_tool(bad, {})
        assert "error" in result
        assert "mcp__<server>__<tool>" in result["error"]
    assert stub.instances[0].calls == []


async def test_call_tool_unknown_server_lists_available(stub):
    client = MCPClient({"srv": {}})

    (result,) = await client.call_tool("mcp__other__echo", {})

    assert "other" in result["error"]
    assert "srv" in result["error"]


async def test_start_is_idempotent(stub):
    client = MCPClient({"a": {}, "b": {}})

    await client.start()
    await client.start()

    assert [instance.started for instance in stub.instances] == [1, 1]


async def test_start_failure_is_isolated(stub):
    client = MCPClient({"good": {}, "broken": {"fail_start": True}})

    await client.start()  # must not raise

    assert [tool.name for tool in await client.get_tools()] == ["mcp__good__echo"]


class GatedAdapter(StubAdapter):
    """An adapter whose ``start`` blocks until the test releases it."""

    instances: list["GatedAdapter"] = []

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.start_calls = 0

    async def start(self) -> None:
        self.start_calls += 1
        self.entered.set()
        await self.release.wait()
        self.started += 1


@pytest.fixture()
def gated(monkeypatch):
    monkeypatch.setattr("cliver_core.mcp._adapters", lambda: (object, GatedAdapter))
    GatedAdapter.instances = []
    return GatedAdapter


async def test_concurrent_start_discovers_once(gated):
    client = MCPClient({"srv": {}})
    adapter = client._adapters["srv"]
    first = asyncio.ensure_future(client.start())
    await adapter.entered.wait()
    second = asyncio.ensure_future(client.start())
    await asyncio.sleep(0)

    adapter.release.set()
    await asyncio.gather(first, second)

    assert adapter.start_calls == 1
    assert [tool.name for tool in await client.get_tools()] == ["mcp__srv__echo"]


async def test_cancelled_start_waiter_does_not_cancel_discovery(gated):
    client = MCPClient({"srv": {}})
    adapter = client._adapters["srv"]
    first = asyncio.ensure_future(client.start())
    await adapter.entered.wait()
    second = asyncio.ensure_future(client.start())
    await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    adapter.release.set()
    await second

    assert adapter.start_calls == 1
    assert [tool.name for tool in await client.get_tools()] == ["mcp__srv__echo"]


async def test_close_cancels_in_flight_start(gated):
    client = MCPClient({"srv": {}})
    adapter = client._adapters["srv"]
    starter = asyncio.ensure_future(client.start())
    await adapter.entered.wait()

    await client.close()

    with pytest.raises(asyncio.CancelledError):
        await starter
    assert adapter.closed == 1
    assert client._started is False


async def test_concurrent_close_closes_each_adapter_once(gated):
    client = MCPClient({"srv": {}})
    adapter = client._adapters["srv"]

    await asyncio.gather(client.close(), client.close())

    assert adapter.closed == 1


async def test_add_after_close_defers_start_until_start(gated):
    client = MCPClient({"srv": {}})
    await client.close()

    await client.add_server("srv", {})
    added = client._adapters["srv"]
    assert added.started == 0  # deferred: the client is not started yet

    starter = asyncio.ensure_future(client.start())
    await added.entered.wait()
    added.release.set()
    await starter

    assert added.started == 1
    assert [tool.name for tool in await client.get_tools()] == ["mcp__srv__echo"]


async def test_get_tools_filters_by_server(stub):
    client = MCPClient({"a": {}, "b": {}})
    await client.start()

    assert [tool.name for tool in await client.get_tools()] == ["mcp__a__echo", "mcp__b__echo"]
    assert [tool.name for tool in await client.get_tools(servers=["b"])] == ["mcp__b__echo"]
    assert await client.get_tools(servers=["nope"]) == []
    # An explicit empty filter means "no servers", not "no filter".
    assert await client.get_tools(servers=[]) == []


async def test_add_remove_and_close(stub):
    client = MCPClient({"a": {}})
    await client.start()
    first = stub.instances[0]

    await client.add_server("a", {})  # replaces → the old adapter is disconnected
    assert first.closed == 1
    replacement = stub.instances[1]
    assert replacement.started == 1
    assert await client.call_tool("mcp__a__echo", {}) == [{"text": "a:echo"}]

    await client.remove_server("a")
    assert replacement.closed == 1
    (result,) = await client.call_tool("mcp__a__echo", {})
    assert "not found" in result["error"]

    await client.close()


async def test_prompts_and_resources_passthrough(stub):
    client = MCPClient({"srv": {}})

    assert await client.list_prompts() == [
        {"server": "srv", "prompts": [{"name": "demo", "description": "A demo prompt", "arguments": []}]}
    ]
    assert await client.list_prompts(servers=["nope"]) == []

    assert await client.get_prompt("srv", "demo", {"topic": "x"}) == [{"messages": [{"text": "demo:{'topic': 'x'}"}]}]
    (missing,) = await client.get_prompt("nope", "demo")
    assert "not found" in missing["error"]

    assert await client.list_resources() == [
        {"server": "srv", "resources": [{"uri": "test://demo", "description": "A demo resource"}]}
    ]
    assert await client.read_resource("srv", "test://demo") == [
        {"contents": [{"uri": "test://demo", "text": "demo body"}]}
    ]
    (missing,) = await client.read_resource("nope", "test://demo")
    assert "not found" in missing["error"]


async def test_from_discovery_builds_a_working_client(stub, tmp_path):
    home = tmp_path / "home"
    (home / ".cliver").mkdir(parents=True)
    (home / ".cliver" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"srv": {"transport": "stdio", "command": "npx"}}}),
        encoding="utf-8",
    )

    client = MCPClient.from_discovery(cwd=tmp_path, home=home, env={})
    await client.start()

    assert [tool.name for tool in await client.get_tools()] == ["mcp__srv__echo"]


def test_empty_client_never_touches_the_adapter_factory(monkeypatch):
    def explode():
        raise AssertionError("the adapter factory must not be called for an empty config")

    monkeypatch.setattr("cliver_core.mcp._adapters", explode)

    client = MCPClient({})

    assert client._adapters == {}
