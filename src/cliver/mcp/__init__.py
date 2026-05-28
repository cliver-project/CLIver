"""MCP client — manages connections to all configured MCP servers.

Replaces the langchain_mcp_adapters-based MCPServersCaller.
Uses the official ``mcp`` Python SDK directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cliver.mcp.adapters import MCPServerAdapter, create_adapter
from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages connections to all configured MCP servers.

    Servers can be added/removed at runtime.  Tools are discovered on startup
    and refreshed when servers are added.

    Usage::

        client = MCPClient({
            "filesystem": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem"]},
            "brave": {"transport": "streamable_http", "url": "http://localhost:8080/mcp"},
        })
        await client.start()
        tools = await client.get_tools()
    """

    def __init__(self, server_configs: dict[str, dict] | None = None):
        self._adapters: dict[str, MCPServerAdapter] = {}
        self._started = False
        for name, cfg in (server_configs or {}).items():
            self._adapters[name] = create_adapter(name, cfg)

    # ── Lifecycle ──────────────────────────────────────────

    async def start(self) -> None:
        """Connect to all configured servers and discover their tools.

        Idempotent — subsequent calls are no-ops.
        """
        if self._started:
            return
        results = await asyncio.gather(
            *(adapter.start() for adapter in self._adapters.values()),
            return_exceptions=True,
        )
        self._started = True
        for name, result in zip(self._adapters.keys(), results):
            if isinstance(result, Exception):
                logger.warning(
                    "MCP server '%s' failed to start: %s", name, result
                )

    async def add_server(self, name: str, config: dict[str, Any]) -> None:
        """Add or replace an MCP server at runtime."""
        if name in self._adapters:
            await self.remove_server(name)
        adapter = create_adapter(name, config)
        await adapter.start()
        self._adapters[name] = adapter

    async def remove_server(self, name: str) -> None:
        """Remove and disconnect an MCP server."""
        adapter = self._adapters.pop(name, None)
        if adapter:
            await adapter.close()

    async def close(self) -> None:
        """Disconnect all servers."""
        self._started = False
        await asyncio.gather(
            *(adapter.close() for adapter in self._adapters.values()),
            return_exceptions=True,
        )
        self._adapters.clear()

    # ── Tools ──────────────────────────────────────────────

    async def get_tools(
        self, servers: list[str] | None = None
    ) -> list[CLIverTool]:
        """Get tools from all servers, or a filtered subset.

        Args:
            servers: If given, only return tools from these server names.
        """
        targets = (
            {s: self._adapters[s] for s in servers if s in self._adapters}
            if servers
            else self._adapters
        )
        tools: list[CLIverTool] = []
        for adapter in targets.values():
            tools.extend(adapter.tools)
        return tools

    async def call_tool(
        self, full_name: str, args: dict[str, Any]
    ) -> list[dict]:
        """Call a tool by its full name: 'server_name#tool_name'.

        Args:
            full_name: Fully qualified tool name with server prefix.
            args: Tool arguments as a dict.

        Returns:
            List of result dicts (may contain 'error' key on failure).
        """
        if "#" not in full_name:
            return [
                {
                    "error": f"'{full_name}' is not an MCP tool (missing server prefix). "
                    "Expected format: 'server_name#tool_name'"
                }
            ]

        server_name, tool_name = full_name.split("#", 1)
        adapter = self._adapters.get(server_name)
        if not adapter:
            return [
                {
                    "error": f"MCP server '{server_name}' not found. "
                    f"Available servers: {list(self._adapters)}"
                }
            ]
        return await adapter.call_tool(tool_name, args)

    # ── Resources ───────────────────────────────────────────

    async def list_resources(
        self, server: str, resource_path: str | None = None
    ) -> list[dict]:
        """List resources from a specific MCP server."""
        adapter = self._adapters.get(server)
        if not adapter:
            return [{"error": f"MCP server '{server}' not found"}]
        return await adapter.list_resources(resource_path)
