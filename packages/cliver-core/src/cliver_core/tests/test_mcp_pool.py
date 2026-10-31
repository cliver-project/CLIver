"""MCP session pool — exclusive leases over streamable-http sessions.

``session_pool_size=N`` caps concurrent tool calls per server.  Idle
sessions are reused; sessions that failed with a raised exception are
closed and replaced.  MCP ``is_error`` results are tool-level, not
transport-level: the session is kept.  Discovery and tool listing stay
ephemeral.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import mcp_types as t
import pytest
from mcp.server.lowlevel import Server

from cliver_core.mcp import adapters as adapters_module
from cliver_core.mcp.adapters import StreamableHTTPAdapter, create_adapter
from cliver_core.mcp.discovery import discover_mcp_servers
from cliver_core.mcp.pool import SessionPool

# ── Pool unit tests (no MCP SDK) ─────────────────────────────


class _FakeSession:
    def __init__(self, serial: int) -> None:
        self.serial = serial
        self.closed = False


def _state() -> dict:
    return {"opened": 0, "fail_next": 0, "active": 0, "peak": 0}


def _factory(state: dict):
    @asynccontextmanager
    async def open_session():
        if state["fail_next"] > 0:
            state["fail_next"] -= 1
            raise ConnectionError("connect failed")
        state["opened"] += 1
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        session = _FakeSession(state["opened"])
        try:
            yield session
        finally:
            state["active"] -= 1
            session.closed = True

    return open_session


@pytest.mark.asyncio
async def test_concurrency_is_capped_and_idle_sessions_are_reused():
    state = _state()
    pool = SessionPool(_factory(state), 2)

    first = await pool.acquire()
    second = await pool.acquire()
    third_task = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)
    assert not third_task.done()  # both leases in use — waiting

    await pool.release(first)
    third = await asyncio.wait_for(third_task, 1)
    assert third is first  # an idle session came back before a new open
    assert state["opened"] == 2
    assert state["peak"] == 2

    await pool.release(second)
    await pool.release(third)
    await pool.close()
    assert first.closed is True
    assert second.closed is True
    assert state["active"] == 0


@pytest.mark.asyncio
async def test_waiters_are_served_in_fifo_order():
    state = _state()
    pool = SessionPool(_factory(state), 1)
    session = await pool.acquire()
    order: list[str] = []

    async def waiter(tag: str):
        lease = await pool.acquire()
        order.append(tag)
        return lease

    first_waiter = asyncio.create_task(waiter("a"))
    await asyncio.sleep(0.01)
    second_waiter = asyncio.create_task(waiter("b"))
    await asyncio.sleep(0.01)

    await pool.release(session)
    lease_a = await asyncio.wait_for(first_waiter, 1)
    await pool.release(lease_a)
    lease_b = await asyncio.wait_for(second_waiter, 1)
    await pool.release(lease_b)

    assert order == ["a", "b"]
    await pool.close()


@pytest.mark.asyncio
async def test_open_failure_propagates_and_frees_the_slot():
    state = _state()
    state["fail_next"] = 1
    pool = SessionPool(_factory(state), 1)

    with pytest.raises(ConnectionError, match="connect failed"):
        await pool.acquire()

    session = await pool.acquire()  # the slot was released on failure
    assert session.serial == 1
    await pool.release(session)
    await pool.close()


@pytest.mark.asyncio
async def test_failed_sessions_are_evicted_and_replaced():
    state = _state()
    pool = SessionPool(_factory(state), 2)

    first = await pool.acquire()
    await pool.release(first, ok=False)

    assert first.closed is True
    replacement = await pool.acquire()
    assert replacement is not first
    assert state["opened"] == 2
    await pool.release(replacement)
    await pool.close()


@pytest.mark.asyncio
async def test_close_closes_leased_sessions_and_is_idempotent():
    state = _state()
    pool = SessionPool(_factory(state), 2)
    first = await pool.acquire()
    second = await pool.acquire()

    await pool.close()
    await pool.close()

    assert first.closed is True
    assert second.closed is True
    assert state["active"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        await pool.acquire()


@pytest.mark.asyncio
async def test_close_wakes_a_waiter_with_runtime_error():
    state = _state()
    pool = SessionPool(_factory(state), 1)
    session = await pool.acquire()
    waiter = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)

    await pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        await waiter
    assert session.closed is True


def test_pool_rejects_another_event_loop():
    holder: dict[str, SessionPool] = {}

    async def make():
        holder["pool"] = SessionPool(_factory(_state()), 1)
        await holder["pool"].close()

    asyncio.run(make())
    pool = holder["pool"]
    with pytest.raises(RuntimeError, match="another event loop"):
        asyncio.run(pool.acquire())


# ── Adapter integration (in-process MCP server) ──────────────


def _make_server() -> Server:
    async def on_list_tools(ctx, params):
        return t.ListToolsResult(
            tools=[
                t.Tool(
                    name="echo",
                    description="Echo",
                    inputSchema={"type": "object", "properties": {"message": {"type": "string"}}},
                ),
                t.Tool(name="slow", description="Slow", inputSchema={"type": "object", "properties": {}}),
                t.Tool(name="boom", description="Always fails", inputSchema={"type": "object", "properties": {}}),
            ]
        )

    async def on_call_tool(ctx, params):
        arguments = params.arguments or {}
        if params.name == "echo":
            return t.CallToolResult(content=[t.TextContent(type="text", text=f"echo: {arguments.get('message', '')}")])
        if params.name == "slow":
            await asyncio.sleep(0.05)
            return t.CallToolResult(content=[t.TextContent(type="text", text="slow done")])
        if params.name == "boom":
            return t.CallToolResult(content=[t.TextContent(type="text", text="kaboom")], isError=True)
        raise ValueError(f"Unknown tool: {params.name}")

    return Server("fake", version="1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


class _PooledAdapter(StreamableHTTPAdapter):
    """StreamableHTTPAdapter whose transport serves an in-process server."""

    def __init__(self, name: str, config: dict, server: Server):
        super().__init__(name, config)
        self._server = server
        self.entered = 0
        self.exited = 0

    @asynccontextmanager
    async def _transport(self):
        self.entered += 1
        try:
            yield self._server
        finally:
            self.exited += 1


def _adapter(*, pool_size: int | None = None, name: str = "fake") -> _PooledAdapter:
    config: dict = {"transport": "streamable-http", "url": "http://in-process/mcp"}
    if pool_size is not None:
        config["session_pool_size"] = pool_size
    return _PooledAdapter(name, config, _make_server())


@pytest.mark.asyncio
async def test_start_stays_ephemeral_and_pool_opens_lazily():
    adapter = _adapter(pool_size=2)

    await adapter.start()

    assert (adapter.entered, adapter.exited) == (1, 1)  # discovery session closed
    assert len(adapter.tools) == 3


@pytest.mark.asyncio
async def test_pooled_calls_cap_concurrency_and_reuse_sessions():
    adapter = _adapter(pool_size=2)
    await adapter.start()

    results = await asyncio.gather(*(adapter.call_tool("slow", {}) for _ in range(3)))

    assert [result[0]["text"] for result in results] == ["slow done"] * 3
    assert adapter.entered - 1 == 2  # two pooled sessions; the third call waited

    await adapter.close()
    assert adapter.exited == adapter.entered


@pytest.mark.asyncio
async def test_sequential_calls_reuse_one_pooled_session():
    adapter = _adapter(pool_size=2)
    await adapter.start()

    await adapter.call_tool("echo", {"message": "a"})
    await adapter.call_tool("echo", {"message": "b"})

    assert adapter.entered == 2  # discovery + one pooled session
    await adapter.close()


@pytest.mark.asyncio
async def test_tool_level_error_keeps_the_session():
    adapter = _adapter(pool_size=1)
    await adapter.start()

    (result,) = await adapter.call_tool("boom", {})
    assert result["error"] == "MCP tool 'boom' failed on server 'fake': kaboom"
    await adapter.call_tool("echo", {"message": "x"})

    assert adapter.entered == 2  # boom did not evict the pooled session
    await adapter.close()


@pytest.mark.asyncio
async def test_raised_failure_evicts_the_session():
    adapter = _adapter(pool_size=1)
    await adapter.start()

    (result,) = await adapter.call_tool("nope", {})
    assert "error" in result
    await adapter.call_tool("echo", {"message": "x"})

    assert adapter.entered == 3  # failed session closed; a replacement opened
    await adapter.close()


@pytest.mark.asyncio
async def test_pooled_error_strings_match_the_ephemeral_path():
    ephemeral = _adapter()
    pooled = _adapter(pool_size=1)
    await ephemeral.start()
    await pooled.start()

    assert await ephemeral.call_tool("boom", {}) == await pooled.call_tool("boom", {})
    (unknown,) = await pooled.call_tool("nope", {})
    assert "'nope'" in unknown["error"] and "'fake'" in unknown["error"]

    await ephemeral.close()
    await pooled.close()


# ── Config validation and discovery ─────────────────────────


def test_create_adapter_rejects_session_pool_size_on_stdio():
    with pytest.raises(ValueError, match="session_pool_size"):
        create_adapter("x", {"transport": "stdio", "command": "npx", "session_pool_size": 2})


@pytest.mark.parametrize("value", [-1, True, "2", 1.5])
def test_create_adapter_rejects_invalid_session_pool_size(value):
    with pytest.raises(ValueError, match="session_pool_size"):
        create_adapter("x", {"transport": "streamable-http", "url": "http://x", "session_pool_size": value})


def test_create_adapter_accepts_zero_as_disabled():
    adapter = create_adapter("x", {"transport": "streamable-http", "url": "http://x", "session_pool_size": 0})
    assert isinstance(adapter, StreamableHTTPAdapter)


def test_discovery_whitelists_session_pool_size_for_streamable_http(tmp_path):
    result = discover_mcp_servers(
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={},
        extra={
            "docs": {"transport": "streamable-http", "url": "https://example/mcp", "session_pool_size": 4},
            "local": {"transport": "stdio", "command": "npx", "session_pool_size": 4},
            "bad": {"transport": "streamable-http", "url": "https://example/mcp", "session_pool_size": "2"},
        },
    )

    assert result.servers["docs"]["session_pool_size"] == 4
    assert "session_pool_size" not in result.servers["local"]
    assert "session_pool_size" not in result.servers["bad"]
    reported = {problem.server for problem in result.problems if problem.code == "invalid-entry"}
    assert {"local", "bad"} <= reported


@pytest.mark.asyncio
async def test_mcp_client_close_reaches_pooled_adapters(monkeypatch):
    from cliver_core.mcp import MCPClient

    adapter = _adapter(pool_size=1)
    monkeypatch.setattr(adapters_module, "create_adapter", lambda name, config: adapter)

    client = MCPClient({"fake": {"transport": "streamable-http", "url": "http://in-process/mcp"}})
    await client.start()
    await adapter.call_tool("echo", {"message": "hi"})
    pooled_open = adapter.entered

    await client.close()

    assert adapter.exited == pooled_open  # the pooled session closed in the manager task
