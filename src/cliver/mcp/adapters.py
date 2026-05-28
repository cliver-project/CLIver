"""MCP transport adapters — one class per transport type.

Each adapter manages the lifecycle of one MCP server connection:
connect → discover tools → serve tool calls → disconnect.
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)


def _normalize_schema(input_schema: dict | None) -> dict:
    """Normalize MCP inputSchema into well-formed JSON Schema."""
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    if not input_schema:
        return schema
    if "properties" in input_schema:
        schema["properties"] = input_schema.get("properties", {})
    if "required" in input_schema:
        schema["required"] = input_schema["required"]
    # Some MCP servers omit "type" on properties — default to string
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict) and "type" not in prop:
            prop["type"] = "string"
    return schema


class MCPServerAdapter(ABC):
    """Manages lifecycle and connection for one MCP server.

    Subclasses implement _connect() for each transport type.
    """

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self._tools: list[CLIverTool] = []
        self._session: ClientSession | None = None
        self._started = False

    @abstractmethod
    @asynccontextmanager
    async def _connect(self):
        """Yield (read_stream, write_stream) for a ClientSession."""
        ...

    async def start(self) -> None:
        """Connect to the server and discover its tools."""
        try:
            async with self._connect() as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    # Note: session becomes invalid after exiting this context.
                    # We store tools but use ephemeral sessions for tool calls.
                    mcp_tools = await session.list_tools()
                    self._tools = [
                        CLIverTool(
                            name=f"{self.name}#{t.name}",
                            description=t.description or f"MCP tool: {t.name}",
                            parameters=_normalize_schema(t.inputSchema),
                            execute=functools.partial(
                                self._call_tool, t.name
                            ),
                        )
                        for t in mcp_tools.tools
                    ]
            self._started = True
            logger.info(
                "MCP server '%s': %d tools discovered",
                self.name,
                len(self._tools),
            )
        except Exception:
            logger.warning(
                "MCP server '%s' failed to start", self.name, exc_info=True
            )

    @property
    def tools(self) -> list[CLIverTool]:
        return list(self._tools)

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> list[dict]:
        """Call a tool on this server using an ephemeral session."""
        try:
            async with self._connect() as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=args)
                    if result.isError:
                        return [
                            {"error": f"MCP tool '{tool_name}' failed on server '{self.name}'"}
                        ]
                    return [c.model_dump() for c in result.content]
        except Exception as e:
            return [{"error": str(e)}]

    async def list_resources(self, resource_path: str | None = None) -> list[dict]:
        """List available resources."""
        try:
            async with self._connect() as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resources = await session.list_resources()
                    uris = [str(r.uri) for r in resources.resources]
                    return [{"resources": uris}]
        except Exception as e:
            return [{"error": str(e)}]

    async def close(self) -> None:
        """Clean up any persistent resources."""
        self._started = False
        self._tools.clear()


class StdioAdapter(MCPServerAdapter):
    """MCP over stdio (subprocess)."""

    @asynccontextmanager
    async def _connect(self):
        async with stdio_client(
            command=self.config["command"],
            args=self.config.get("args", []),
            env=self.config.get("env"),
        ) as (read, write):
            yield read, write


class SSEAdapter(MCPServerAdapter):
    """MCP over Server-Sent Events."""

    @asynccontextmanager
    async def _connect(self):
        async with sse_client(
            url=self.config["url"],
            headers=self.config.get("headers"),
        ) as (read, write):
            yield read, write


class StreamableHTTPAdapter(MCPServerAdapter):
    """MCP over Streamable HTTP."""

    @asynccontextmanager
    async def _connect(self):
        async with streamable_http_client(
            url=self.config["url"],
            headers=self.config.get("headers"),
        ) as (read, write):
            yield read, write


_TRANSPORTS: dict[str, type[MCPServerAdapter]] = {
    "stdio": StdioAdapter,
    "sse": SSEAdapter,
    "streamable_http": StreamableHTTPAdapter,
}


def create_adapter(name: str, config: dict[str, Any]) -> MCPServerAdapter:
    transport = config.get("transport", "stdio")
    cls = _TRANSPORTS.get(transport)
    if not cls:
        raise ValueError(
            f"Unknown MCP transport '{transport}' for server '{name}'. "
            f"Supported: {list(_TRANSPORTS)}"
        )
    return cls(name=name, config=config)
