"""MCP client — manages connections to all configured MCP servers.

Servers come from cliver's own places via :func:`discover_mcp_servers`
(``~/.cliver/mcp.json`` + ``~/.cliver/mcps/*.json`` + the project-local
``.cliver/mcp.json``; discovery is a pure read — nothing is persisted), or
from configs the caller passes in directly.  Model-facing tools are named
``mcp__<server>__<tool>``; prompts and resources are listed/read on demand
with their descriptions and argument metadata (never advertised as tools).

Usage::

    from cliver_core.mcp import MCPClient, discover_mcp_servers

    result = discover_mcp_servers()                  # cliver's own places
    # opt into more places:
    result = discover_mcp_servers(extra_paths=["~/team-mcps/", "extra.json"])
    client = MCPClient(result.servers)
    await client.start()
    tools = await client.get_tools()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cliver_core.mcp.discovery import (
    MCPDiscoveryProblem,
    MCPDiscoveryResult,
    MCPServerProvenance,
    discover_mcp_servers,
)
from cliver_core.tool import CLIverTool

__all__ = [
    "MCPClient",
    "MCPDiscoveryProblem",
    "MCPDiscoveryResult",
    "MCPServerProvenance",
    "discover_mcp_servers",
]

_TOOL_PREFIX = "mcp__"


def _adapters():
    """Lazy import of MCP adapters — keeps the mcp SDK out of `import cliver_core`."""
    try:
        from cliver_core.mcp.adapters import MCPServerAdapter, create_adapter
    except ModuleNotFoundError as exc:
        guarded = {
            "mcp",
            "mcp_types",
            "httpx2",
            "httpcore2",
            "starlette",
            "sse_starlette",
            "uvicorn",
            "jsonschema",
            "pyjwt",
            "cryptography",
        }
        if exc.name in guarded or (exc.name or "").startswith("mcp."):
            raise ImportError(
                "The 'mcp' package is missing; it is a cliver-core dependency — reinstall cliver-core"
            ) from exc
        raise
    return MCPServerAdapter, create_adapter


logger = logging.getLogger(__name__)


class MCPClient:
    """Manages connections to all configured MCP servers.

    Servers can be added/removed at runtime.  Tools are discovered on
    ``start()`` and refreshed when servers are added; tool results always
    come back as lists of dicts (an ``{"error": ...}`` item on failure).

    Usage::

        client = MCPClient({
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            },
            "docs": {"transport": "streamable-http", "url": "https://example/mcp"},
        })
        await client.start()
        tools = await client.get_tools()
    """

    def __init__(self, server_configs: dict[str, dict] | None = None):
        self._adapters: dict = {}
        self._started = False
        if server_configs:
            _, create_adapter = _adapters()
            for name, cfg in server_configs.items():
                self._adapters[name] = create_adapter(name, cfg)

    @classmethod
    def from_discovery(cls, *, extra: dict | None = None, **discovery_kwargs: Any) -> "MCPClient":
        """Build a client from cliver-local discovery (``discover_mcp_servers``).

        Extra keyword arguments go straight to ``discover_mcp_servers`` —
        ``extra_paths`` is the opt-in for places beyond cliver's own.
        """
        return cls(discover_mcp_servers(extra=extra, **discovery_kwargs).servers)

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
        for name, result in zip(self._adapters.keys(), results, strict=False):
            if isinstance(result, Exception):
                logger.warning("MCP server '%s' failed to start: %s", name, result)

    async def add_server(self, name: str, config: dict[str, Any]) -> None:
        """Add or replace an MCP server at runtime."""
        if name in self._adapters:
            await self.remove_server(name)
        _, create_adapter = _adapters()
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

    async def get_tools(self, servers: list[str] | None = None) -> list[CLIverTool]:
        """Get tools from all servers, or a filtered subset.

        Args:
            servers: If given, only return tools from these server names.
        """
        targets = {s: self._adapters[s] for s in servers if s in self._adapters} if servers else self._adapters
        tools: list[CLIverTool] = []
        for adapter in targets.values():
            tools.extend(adapter.tools)
        return tools

    async def call_tool(self, full_name: str, args: dict[str, Any]) -> list[dict]:
        """Call a tool by its full name: ``mcp__<server>__<tool>``.

        Args:
            full_name: Fully qualified tool name with the ``mcp__`` prefix.
            args: Tool arguments as a dict.

        Returns:
            List of result dicts (may contain 'error' key on failure).
        """
        if not full_name.startswith(_TOOL_PREFIX):
            return [
                {
                    "error": f"'{full_name}' is not an MCP tool (missing 'mcp__' prefix). "
                    "Expected format: 'mcp__<server>__<tool>'"
                }
            ]

        server_name, separator, tool_name = full_name[len(_TOOL_PREFIX) :].partition("__")
        if not separator or not server_name or not tool_name:
            return [{"error": f"'{full_name}' is not a valid MCP tool name. Expected format: 'mcp__<server>__<tool>'"}]
        adapter = self._adapters.get(server_name)
        if not adapter:
            return [{"error": f"MCP server '{server_name}' not found. Available servers: {list(self._adapters)}"}]
        return await adapter.call_tool(tool_name, args)

    # ── Prompts & resources (reserved — not model-facing) ──

    def _resolve(self, servers: list[str] | None) -> list[tuple[str, Any]]:
        if servers is None:
            return list(self._adapters.items())
        return [(name, self._adapters[name]) for name in servers if name in self._adapters]

    async def list_prompts(self, servers: list[str] | None = None) -> list[dict]:
        """List prompts (names, descriptions, arguments) from the given servers."""
        results: list[dict] = []
        for name, adapter in self._resolve(servers):
            for item in await adapter.list_prompts():
                results.append({"server": name, **item})
        return results

    async def get_prompt(self, server: str, name: str, arguments: dict[str, str] | None = None) -> list[dict]:
        """Render one prompt on one server (prompts take arguments)."""
        adapter = self._adapters.get(server)
        if not adapter:
            return [{"error": f"MCP server '{server}' not found. Available servers: {list(self._adapters)}"}]
        return await adapter.get_prompt(name, arguments)

    async def list_resources(self, servers: list[str] | None = None) -> list[dict]:
        """List resources (full records: uri, name, description, mime type, …)."""
        results: list[dict] = []
        for name, adapter in self._resolve(servers):
            for item in await adapter.list_resources():
                results.append({"server": name, **item})
        return results

    async def read_resource(self, server: str, uri: str) -> list[dict]:
        """Read one resource by URI from one server."""
        adapter = self._adapters.get(server)
        if not adapter:
            return [{"error": f"MCP server '{server}' not found. Available servers: {list(self._adapters)}"}]
        return await adapter.read_resource(uri)
