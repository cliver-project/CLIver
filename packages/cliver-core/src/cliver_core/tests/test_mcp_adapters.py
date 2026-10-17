# ruff: noqa: E402 — the SDK guard below must run before the mcp imports.
"""MCP adapters against the real mcp SDK — offline, in-process only.

Two transport shapes are exercised: ``Client(Server instance)`` (the SDK's
in-process direct dispatcher) and real wire streams over
``create_client_server_memory_streams()`` (which pins the negotiated protocol
version).  No subprocesses and no network.  The MCP SDK is an optional extra,
so this module skips (via ``importorskip``) when it is not installed.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("mcp")

import anyio
import mcp_types as t
from mcp import StdioServerParameters
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_client_server_memory_streams

from cliver_core.mcp import adapters as adapters_module
from cliver_core.mcp.adapters import (
    CLIENT_INFO,
    MCPServerAdapter,
    StdioAdapter,
    StreamableHTTPAdapter,
    _normalize_schema,
    create_adapter,
)


def _make_server() -> Server:
    """A lowlevel server with the shapes the adapters must handle."""

    async def on_list_tools(ctx, params):
        return t.ListToolsResult(
            tools=[
                t.Tool(
                    name="echo",
                    description="Echo text",
                    inputSchema={
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                ),
                t.Tool(name="loose", inputSchema={"type": "object", "properties": {"q": {}}}),
                t.Tool(name="boom", description="Always fails", inputSchema={"type": "object", "properties": {}}),
                t.Tool(name="bad__name", description="Skipped", inputSchema={"type": "object", "properties": {}}),
            ]
        )

    async def on_call_tool(ctx, params):
        if params.name == "echo":
            arguments = params.arguments or {}
            return t.CallToolResult(content=[t.TextContent(type="text", text=f"echo: {arguments.get('message', '')}")])
        if params.name == "boom":
            return t.CallToolResult(content=[t.TextContent(type="text", text="kaboom")], isError=True)
        if params.name == "structured":
            # Deliberately unlisted: exercises the empty-content fallback
            # without the SDK's structured-output validation.
            return t.CallToolResult(content=[], structuredContent={"answer": 42})
        raise ValueError(f"Unknown tool: {params.name}")

    async def on_list_prompts(ctx, params):
        return t.ListPromptsResult(
            prompts=[
                t.Prompt(
                    name="greet",
                    description="Greeting",
                    arguments=[t.PromptArgument(name="who", description="Whom", required=True)],
                )
            ]
        )

    async def on_get_prompt(ctx, params):
        who = (params.arguments or {}).get("who", "world")
        return t.GetPromptResult(
            description="Greeting",
            messages=[t.PromptMessage(role="user", content=t.TextContent(type="text", text=f"Hello {who}"))],
        )

    async def on_list_resources(ctx, params):
        return t.ListResourcesResult(
            resources=[t.Resource(uri="test://demo", name="demo", description="A demo resource", mimeType="text/plain")]
        )

    async def on_read_resource(ctx, params):
        return t.ReadResourceResult(
            contents=[t.TextResourceContents(uri="test://demo", mimeType="text/plain", text="demo body")]
        )

    return Server(
        "fake",
        version="1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_prompts=on_list_prompts,
        on_get_prompt=on_get_prompt,
        on_list_resources=on_list_resources,
        on_read_resource=on_read_resource,
    )


@pytest.fixture()
def server() -> Server:
    return _make_server()


class _InProcessAdapter(MCPServerAdapter):
    """Serves ``Client(Server instance)`` — the SDK's in-process path."""

    transport = "stdio"

    def __init__(self, name: str, config: dict, server: Server):
        super().__init__(name, config)
        self._server = server

    @asynccontextmanager
    async def _transport(self):
        yield self._server


@asynccontextmanager
async def _streams_cm(streams):
    yield streams


class _MemoryStreamAdapter(MCPServerAdapter):
    """Runs the server over real wire streams (in-memory transport)."""

    transport = "stdio"

    def __init__(self, name: str, config: dict, server: Server):
        super().__init__(name, config)
        self._server = server

    @asynccontextmanager
    async def _transport(self):
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    self._server.run,
                    server_streams[0],
                    server_streams[1],
                    self._server.create_initialization_options(),
                )
                try:
                    yield _streams_cm(client_streams)
                finally:
                    task_group.cancel_scope.cancel()


# ── Tool discovery and naming ─────────────────────────────


async def test_start_discovers_tools_with_the_mcp_prefix(server):
    adapter = _InProcessAdapter("fake", {}, server)

    await adapter.start()

    tools = adapter.tools
    assert [tool.name for tool in tools] == [
        "mcp__fake__echo",
        "mcp__fake__loose",
        "mcp__fake__boom",
    ]  # bad__name skipped
    assert tools[0].description == "Echo text"
    assert tools[1].description == "MCP tool: loose"  # default description
    assert tools[0].parameters == {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }
    assert tools[1].parameters["properties"]["q"] == {"type": "string"}  # missing type defaulted
    assert callable(tools[0].execute)


async def test_enabled_and_disabled_tool_filters(server):
    enabled = _InProcessAdapter("fake", {"enabled_tools": ["echo", "boom"]}, server)
    await enabled.start()
    assert [tool.name for tool in enabled.tools] == ["mcp__fake__echo", "mcp__fake__boom"]

    disabled = _InProcessAdapter("fake", {"disabled_tools": ["echo"]}, _make_server())
    await disabled.start()
    assert [tool.name for tool in disabled.tools] == ["mcp__fake__loose", "mcp__fake__boom"]


# ── Tool calls ────────────────────────────────────────────


async def test_call_tool_returns_content_blocks_without_none_keys(server):
    adapter = _InProcessAdapter("fake", {}, server)
    await adapter.start()

    result = await adapter.call_tool("echo", {"message": "hi"})

    assert result == [{"type": "text", "text": "echo: hi"}]


async def test_call_tool_error_result_names_tool_and_server(server):
    adapter = _InProcessAdapter("fake", {}, server)

    (result,) = await adapter.call_tool("boom", {})

    assert result["error"] == "MCP tool 'boom' failed on server 'fake': kaboom"


async def test_call_tool_unknown_tool_names_tool_and_server(server):
    adapter = _InProcessAdapter("fake", {}, server)

    (result,) = await adapter.call_tool("nope", {})

    assert "error" in result
    assert "'nope'" in result["error"] and "'fake'" in result["error"]


async def test_call_tool_structured_content_fallback(server):
    adapter = _InProcessAdapter("fake", {}, server)

    result = await adapter.call_tool("structured", {})

    assert result == [{"text": '{"answer": 42}'}]


# ── Prompts and resources (reserved surfaces) ─────────────


async def test_prompts_and_resources_round_trip(server):
    adapter = _InProcessAdapter("fake", {}, server)

    (prompts,) = await adapter.list_prompts()
    assert prompts == {
        "prompts": [
            {
                "name": "greet",
                "description": "Greeting",
                "arguments": [{"name": "who", "description": "Whom", "required": True}],
            }
        ]
    }

    (rendered,) = await adapter.get_prompt("greet", {"who": "cliver"})
    assert rendered["messages"] == [{"role": "user", "content": {"type": "text", "text": "Hello cliver"}}]

    (resources,) = await adapter.list_resources()
    assert resources == {
        "resources": [
            {"uri": "test://demo", "name": "demo", "description": "A demo resource", "mime_type": "text/plain"}
        ]
    }

    (contents,) = await adapter.read_resource("test://demo")
    assert contents == {"contents": [{"uri": "test://demo", "mime_type": "text/plain", "text": "demo body"}]}


# ── Wire-level protocol negotiation ───────────────────────


@pytest.mark.parametrize(("mode", "expected"), [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
async def test_wire_protocol_negotiation(mode, expected):
    server = _make_server()
    adapter = _MemoryStreamAdapter("wire", {"mode": mode}, server)

    async with adapter.session() as client:
        assert client.protocol_version == expected

    await adapter.start()
    assert [tool.name for tool in adapter.tools] == ["mcp__wire__echo", "mcp__wire__loose", "mcp__wire__boom"]
    assert await adapter.call_tool("echo", {"message": "wire"}) == [{"type": "text", "text": "echo: wire"}]


# ── Failure isolation ─────────────────────────────────────


async def test_transport_failure_leaves_zero_tools_without_raising():
    class Broken(MCPServerAdapter):
        transport = "stdio"

        @asynccontextmanager
        async def _transport(self):
            raise RuntimeError("cannot connect")
            yield None  # pragma: no cover

    adapter = Broken("broken", {})

    await adapter.start()

    assert adapter.tools == []
    (result,) = await adapter.call_tool("anything", {})
    assert "error" in result and "'broken'" in result["error"]


# ── Transport construction (recording stubs — no I/O) ─────


class _RecordingClient:
    """Stands in for ``mcp.Client`` — records construction kwargs, no I/O."""

    records: list[dict] = []

    def __init__(self, server, *, mode="auto", read_timeout_seconds=None, client_info=None):
        _RecordingClient.records.append(
            {"server": server, "mode": mode, "timeout": read_timeout_seconds, "client_info": client_info}
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture()
def recorded_client(monkeypatch):
    _RecordingClient.records = []
    monkeypatch.setattr(adapters_module, "Client", _RecordingClient)
    return _RecordingClient


@asynccontextmanager
async def _null_transport():
    yield ("read", "write")


async def test_stdio_transport_builds_server_parameters(recorded_client):
    adapter = StdioAdapter(
        "s",
        {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "pkg", 5],
            "env": {"K": 1},
            "cwd": "/tmp",
            "timeout": 5,
            "mode": "legacy",
        },
    )

    async with adapter.session():
        pass

    (record,) = recorded_client.records
    params = record["server"]
    assert isinstance(params, StdioServerParameters)
    assert params.command == "npx"
    assert params.args == ["-y", "pkg", "5"]
    assert params.env == {"K": "1"}
    assert params.cwd == "/tmp"
    assert record["mode"] == "legacy"
    assert record["timeout"] == 5.0
    assert record["client_info"] is CLIENT_INFO


async def test_streamable_http_without_headers_lets_the_sdk_own_the_client(monkeypatch, recorded_client):
    calls = []

    def fake_transport(url, *, http_client=None, terminate_on_close=True):
        calls.append({"url": url, "http_client": http_client, "terminate_on_close": terminate_on_close})
        return _null_transport()

    monkeypatch.setattr(adapters_module, "streamable_http_client", fake_transport)
    adapter = StreamableHTTPAdapter("h", {"transport": "streamable-http", "url": "https://x/mcp"})

    async with adapter.session():
        pass

    assert calls == [{"url": "https://x/mcp", "http_client": None, "terminate_on_close": True}]


async def test_streamable_http_headers_ride_a_caller_owned_client(monkeypatch, recorded_client):
    created, events, calls = [], [], []

    class RecordingHTTPClient:
        def __init__(self, *, headers, timeout):
            created.append(self)
            self.headers = headers
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            events.append("closed")
            return False

    def fake_transport(url, *, http_client=None, terminate_on_close=True):
        calls.append({"url": url, "http_client": http_client, "terminate_on_close": terminate_on_close})
        return _null_transport()

    monkeypatch.setattr(adapters_module.httpx2, "AsyncClient", RecordingHTTPClient)
    monkeypatch.setattr(adapters_module, "streamable_http_client", fake_transport)
    adapter = StreamableHTTPAdapter(
        "h",
        {"transport": "streamable-http", "url": "https://x/mcp", "headers": {"Authorization": "Bearer tok"}},
    )

    async with adapter.session():
        events.append("running")

    assert created[0].headers == {"Authorization": "Bearer tok"}
    assert calls[0]["http_client"] is created[0]  # the client we created is the one the transport gets
    assert events == ["running", "closed"]  # closed only after the session ended


# ── Construction and schema normalization ─────────────────


def test_create_adapter_validates_transport_and_required_keys():
    assert isinstance(create_adapter("s", {"transport": "stdio", "command": "npx"}), StdioAdapter)
    assert isinstance(
        create_adapter("h", {"transport": "streamable-http", "url": "https://x/mcp"}), StreamableHTTPAdapter
    )

    with pytest.raises(ValueError, match=re.escape("Supported: ['stdio', 'streamable-http']")):
        create_adapter("s", {"transport": "sse"})

    with pytest.raises(ValueError, match="requires command"):
        create_adapter("s", {"transport": "stdio"})

    with pytest.raises(ValueError, match="requires url"):
        create_adapter("h", {"transport": "streamable-http"})


def test_startup_timeout_defaults_and_overrides():
    assert StdioAdapter("s", {"command": "npx"})._startup_timeout() == 60.0
    assert StreamableHTTPAdapter("h", {"url": "https://x/mcp"})._startup_timeout() == 30.0
    assert StdioAdapter("s", {"command": "npx", "startup_timeout": 5})._startup_timeout() == 5.0
    assert StdioAdapter("s", {"command": "npx", "startup_timeout": None})._startup_timeout() is None


def test_normalize_schema_preserves_refs_and_copies():
    source = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}, "raw": {}},
        "required": ["item"],
        "$defs": {"Item": {"type": "string"}},
        "additionalProperties": False,
    }

    schema = _normalize_schema(source)

    assert schema["$defs"] == {"Item": {"type": "string"}}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["item"] == {"$ref": "#/$defs/Item"}  # no type next to a $ref
    assert schema["properties"]["raw"] == {"type": "string"}
    assert source["properties"]["raw"] == {}  # the server's dict is untouched
    assert _normalize_schema(None) == {"type": "object", "properties": {}}
