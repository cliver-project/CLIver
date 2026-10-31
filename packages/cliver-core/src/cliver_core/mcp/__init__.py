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
import contextlib
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
        # Single-flight discovery: every concurrent caller awaits one shared
        # task, so adapters are started once.  ``_close_task`` makes close
        # idempotent and cancellable from any caller.
        self._start_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
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

        Idempotent and single-flight: concurrent callers share one discovery
        pass.  Cancelling one caller does not cancel the discovery for the
        others (the shared task is shielded); cancelling the last caller
        leaves discovery running to completion in the background.
        """
        if self._started:
            return
        if self._start_task is None:
            self._start_task = asyncio.ensure_future(self._start_all())
        await asyncio.shield(self._start_task)

    async def _start_all(self) -> None:
        """Start every adapter once and record per-server failures."""
        names = list(self._adapters)
        try:
            results = await asyncio.gather(
                *(self._adapters[name].start() for name in names),
                return_exceptions=True,
            )
            self._started = True
            for name, result in zip(names, results, strict=False):
                if isinstance(result, Exception):
                    logger.warning("MCP server '%s' failed to start: %s", name, result)
        finally:
            self._start_task = None

    async def _await_start(self) -> None:
        """Wait for an in-flight discovery before mutating the server table."""
        task = self._start_task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(task)

    async def add_server(self, name: str, config: dict[str, Any]) -> None:
        """Add or replace an MCP server at runtime.

        If the client is already serving, the new adapter is started
        immediately; otherwise it is started by the next :meth:`start`.
        """
        await self._await_start()
        if name in self._adapters:
            await self.remove_server(name)
        _, create_adapter = _adapters()
        adapter = create_adapter(name, config)
        if self._started:
            await adapter.start()
        self._adapters[name] = adapter

    async def remove_server(self, name: str) -> None:
        """Remove and disconnect an MCP server."""
        await self._await_start()
        adapter = self._adapters.pop(name, None)
        if adapter:
            await adapter.close()

    async def close(self) -> None:
        """Cancel an in-flight start, then disconnect all servers.

        Idempotent and safe to call concurrently: every caller awaits the
        same shutdown pass.  A start() caller that was waiting on the
        cancelled discovery sees ``CancelledError``.
        """
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._close_all())
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        task = self._start_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._start_task = None
        self._started = False
        adapters = list(self._adapters.values())
        self._adapters.clear()
        if adapters:
            await asyncio.gather(
                *(adapter.close() for adapter in adapters),
                return_exceptions=True,
            )
        self._close_task = None

    # ── Tools ──────────────────────────────────────────────

    async def get_tools(self, servers: list[str] | None = None) -> list[CLIverTool]:
        """Get tools from all servers, or a filtered subset.

        Args:
            servers: If given, only return tools from these server names.
                ``None`` means every server; an empty list means none.
        """
        targets = (
            {s: self._adapters[s] for s in servers if s in self._adapters} if servers is not None else self._adapters
        )
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
