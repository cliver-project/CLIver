"""MCP transport adapters — exactly two transports: ``stdio`` and
``streamable-http``.

Each adapter runs ephemeral sessions on the high-level ``mcp.Client``
(protocol ``2026-07-28`` via ``mode="auto"``, with the legacy ``initialize``
handshake as fallback): connect → list/call → disconnect.  Nothing is kept
open between operations.

Tools are model-facing and named ``mcp__<server>__<tool>`` — neither
component may contain ``__`` (a tool whose name does is skipped).
Prompts and resources are reserved: listed and read on demand with their
descriptions and argument metadata, never advertised as tools.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import metadata
from typing import Any, AsyncIterator

import httpx2
from mcp import Client, Implementation, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from cliver_core.mcp.pool import SessionPool
from cliver_core.timeouts import timeout_after
from cliver_core.tool import CLIverTool

logger = logging.getLogger(__name__)

_DEFAULT_STARTUP_TIMEOUT_STDIO = 60.0
_DEFAULT_STARTUP_TIMEOUT_REMOTE = 30.0
_MAX_LIST_PAGES = 50
_MAX_TOOLS_WARN = 200
_TOOL_PREFIX = "mcp__"
_ERROR_DETAIL_LIMIT = 300


@asynccontextmanager
async def _timeout_after(seconds: float) -> AsyncIterator[None]:
    """MCP connection timeout — the shared cancel-based helper with its message."""
    async with timeout_after(seconds, message=f"startup timed out after {seconds:g}s"):
        yield


_SCHEMA_PASSTHROUGH = ("$defs", "definitions", "additionalProperties", "title")

try:
    _VERSION = metadata.version("cliver-core")
except metadata.PackageNotFoundError:  # pragma: no cover — source checkout without install
    _VERSION = "0.0.0"

CLIENT_INFO = Implementation(name="cliver-core", version=_VERSION)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_pool_size(name: str, config: dict[str, Any], *, supported: bool) -> int:
    """``session_pool_size`` as an int — 0 (or absent/None) means no pool."""
    value = config.get("session_pool_size")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"MCP server '{name}': session_pool_size must be a non-negative integer (got {value!r})")
    if value and not supported:
        raise ValueError(f"MCP server '{name}': session_pool_size is supported for streamable-http only")
    return value


def _dump(model: Any) -> dict:
    """Pydantic record → JSON-friendly dict (URLs as strings, no None keys)."""
    return model.model_dump(mode="json", exclude_none=True)


def _error_detail(result: Any) -> str:
    """First text block of a failed tool result, truncated — never the whole payload."""
    for block in result.content or ():
        text = getattr(block, "text", None)
        if text:
            return text[:_ERROR_DETAIL_LIMIT]
    return "no error detail"


def _normalize_schema(input_schema: dict | None) -> dict:
    """Copy an MCP inputSchema into a well-formed JSON Schema.

    The server's dict is never mutated; ``$defs``/``definitions``/
    ``additionalProperties``/``title`` are preserved so ``$ref`` schemas
    survive, and properties without a ``type`` (or ``$ref``) default to
    string.
    """
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    if not isinstance(input_schema, dict):
        return schema
    for key in _SCHEMA_PASSTHROUGH:
        if key in input_schema:
            schema[key] = copy.deepcopy(input_schema[key])
    properties = input_schema.get("properties")
    if isinstance(properties, dict):
        schema["properties"] = copy.deepcopy(properties)
    required = input_schema.get("required")
    if isinstance(required, list):
        schema["required"] = list(required)
    for prop in schema["properties"].values():
        if isinstance(prop, dict) and "type" not in prop and "$ref" not in prop:
            prop["type"] = "string"
    return schema


class MCPServerAdapter(ABC):
    """Lifecycle and sessions for one MCP server (one transport).

    ``session()`` opens an ephemeral session — connect, use, disconnect.
    Streamable-http adapters may instead enable a session pool with
    ``session_pool_size=N`` (max concurrent tool calls; sessions are reused
    and an idle call never dials again).  Prompts, resources and ``start()``
    discovery always stay ephemeral.
    """

    transport: str = ""
    _required: tuple[str, ...] = ()
    #: Whether ``session_pool_size`` applies — streamable-http only.
    supports_session_pool: bool = False

    def __init__(self, name: str, config: dict[str, Any]):
        missing = [key for key in self._required if not config.get(key)]
        if missing:
            raise ValueError(f"MCP server '{name}': {self.transport} transport requires {', '.join(missing)}")
        self.name = name
        self.config = config
        self._pool_size = _validate_pool_size(name, config, supported=self.supports_session_pool)
        self._pool: SessionPool | None = None
        self._tools: list[CLIverTool] = []

    @abstractmethod
    @asynccontextmanager
    async def _transport(self) -> AsyncIterator[Any]:
        """Yield the transport object for ``Client(server=...)``; owns its resources."""
        ...

    def _startup_timeout(self) -> float | None:
        if "startup_timeout" in self.config:
            return _as_float(self.config["startup_timeout"])
        if self.transport == "stdio":
            return _DEFAULT_STARTUP_TIMEOUT_STDIO
        return _DEFAULT_STARTUP_TIMEOUT_REMOTE

    @asynccontextmanager
    async def _connected(self) -> AsyncIterator[Client]:
        """One connected ``Client`` — transport, connect, yield, disconnect.

        The caller (ephemeral path or the pool's manager task) owns the
        lifecycle; every enter/exit happens in the calling task.
        """
        async with self._transport() as server_arg:
            client = Client(
                server_arg,
                mode=self.config.get("mode", "auto"),
                read_timeout_seconds=_as_float(self.config.get("timeout")),
                client_info=CLIENT_INFO,
            )
            stack = AsyncExitStack()
            try:
                startup = self._startup_timeout()
                if startup is None:
                    await stack.enter_async_context(client)
                else:
                    async with _timeout_after(startup):
                        await stack.enter_async_context(client)
                yield client
            finally:
                await stack.aclose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Client]:
        """Open one ephemeral, connected ``Client`` session."""
        async with self._connected() as client:
            yield client

    def _ensure_pool(self) -> SessionPool:
        """The lazy session pool — created on first pooled call, caller's loop."""
        if self._pool is None:
            self._pool = SessionPool(self._connected, self._pool_size)
        return self._pool

    async def _list_all(self, client: Client, method: str, field: str) -> list[Any]:
        items: list[Any] = []
        cursor: str | None = None
        for _ in range(_MAX_LIST_PAGES):
            result = await getattr(client, method)(cursor=cursor)
            items.extend(getattr(result, field))
            cursor = result.next_cursor
            if not cursor:
                break
        return items

    # ── Tools (model-facing) ───────────────────────────────

    async def start(self) -> None:
        """Connect once and discover this server's tools (in-memory)."""
        try:
            async with self.session() as client:
                mcp_tools = await self._list_all(client, "list_tools", "tools")
                self._tools = self._build_tools(mcp_tools)
                protocol = client.protocol_version
            logger.info("MCP server '%s': %d tools discovered (protocol %s)", self.name, len(self._tools), protocol)
        except Exception as exc:
            logger.warning("MCP server '%s' failed to start: %s", self.name, exc)
            logger.debug("MCP server '%s' start failure detail", self.name, exc_info=True)

    def _build_tools(self, mcp_tools: list[Any]) -> list[CLIverTool]:
        enabled = self.config.get("enabled_tools")
        disabled = set(self.config.get("disabled_tools") or ())
        tools: list[CLIverTool] = []
        seen: set[str] = set()
        for mcp_tool in mcp_tools:
            if "__" in mcp_tool.name:
                logger.warning(
                    "MCP server '%s': skipping tool '%s' — names may not contain '__' (mcp__<server>__<tool> contract)",
                    self.name,
                    mcp_tool.name,
                )
                continue
            if mcp_tool.name in seen:
                continue
            seen.add(mcp_tool.name)
            if enabled and mcp_tool.name not in enabled:
                continue
            if mcp_tool.name in disabled:
                continue
            tools.append(
                CLIverTool(
                    name=f"{_TOOL_PREFIX}{self.name}__{mcp_tool.name}",
                    description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                    parameters=_normalize_schema(mcp_tool.input_schema),
                    execute=functools.partial(self._call_tool, mcp_tool.name),
                    # Annotations are self-declared hints (MCP spec): they set
                    # the default category, never a permission boundary.
                    # (SDK attribute is snake_case read_only_hint.)
                    safety="safe" if getattr(mcp_tool.annotations, "read_only_hint", None) else "mutating",
                    # Identity routing: AgentCore dispatches via the MCP client
                    # because of this origin, not because of the name prefix.
                    origin="mcp",
                )
            )
        if len(tools) > _MAX_TOOLS_WARN:
            logger.warning(
                "MCP server '%s' exposes %d tools — consider enabled_tools/disabled_tools",
                self.name,
                len(tools),
            )
        return tools

    def _format_result(self, result: Any, tool_name: str) -> list[dict]:
        """SDK call result → content blocks, or the server-named error entry."""
        if result.is_error:
            return [{"error": f"MCP tool '{tool_name}' failed on server '{self.name}': {_error_detail(result)}"}]
        content = [_dump(block) for block in result.content]
        if not content and result.structured_content is not None:
            content = [{"text": json.dumps(result.structured_content, ensure_ascii=False, default=str)}]
        return content

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> list[dict]:
        """Call a tool on this server — pooled when enabled, ephemeral otherwise.

        A raised failure releases the pooled session for replacement (never
        retried here); an MCP ``is_error`` result is tool-level and keeps the
        session.
        """
        if self._pool_size:
            return await self._call_tool_pooled(tool_name, args)
        try:
            async with self.session() as client:
                result = await client.call_tool(tool_name, args)
            return self._format_result(result, tool_name)
        except Exception as exc:
            return [{"error": f"MCP tool '{tool_name}' on server '{self.name}' failed: {exc}"}]

    async def _call_tool_pooled(self, tool_name: str, args: dict[str, Any]) -> list[dict]:
        pool = self._ensure_pool()
        try:
            session = await pool.acquire()
        except Exception as exc:
            return [{"error": f"MCP tool '{tool_name}' on server '{self.name}' failed: {exc}"}]
        ok = False
        try:
            result = await session.call_tool(tool_name, args)
            ok = True
            return self._format_result(result, tool_name)
        except Exception as exc:
            return [{"error": f"MCP tool '{tool_name}' on server '{self.name}' failed: {exc}"}]
        finally:
            await pool.release(session, ok=ok)

    def _call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Executor bound into ``CLIverTool.execute`` — returns a coroutine to await."""
        return self.call_tool(tool_name, kwargs)

    # ── Prompts & resources (reserved — listed/read on demand) ──

    async def list_prompts(self) -> list[dict]:
        """List this server's prompts — names, descriptions, argument metadata."""
        try:
            async with self.session() as client:
                prompts = await self._list_all(client, "list_prompts", "prompts")
            return [{"prompts": [_dump(prompt) for prompt in prompts]}]
        except Exception as exc:
            return [{"error": f"MCP server '{self.name}' failed to list prompts: {exc}"}]

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> list[dict]:
        """Render a prompt (prompts take arguments for dynamic responses)."""
        try:
            async with self.session() as client:
                result = await client.get_prompt(name, arguments)
            return [_dump(result)]
        except Exception as exc:
            return [{"error": f"MCP prompt '{name}' on server '{self.name}' failed: {exc}"}]

    async def list_resources(self) -> list[dict]:
        """List this server's resources — full records, descriptions included."""
        try:
            async with self.session() as client:
                resources = await self._list_all(client, "list_resources", "resources")
            return [{"resources": [_dump(resource) for resource in resources]}]
        except Exception as exc:
            return [{"error": f"MCP server '{self.name}' failed to list resources: {exc}"}]

    async def read_resource(self, uri: str) -> list[dict]:
        """Read one resource by URI (text or base64-encoded blob)."""
        try:
            async with self.session() as client:
                result = await client.read_resource(uri)
            return [{"contents": [_dump(contents) for contents in result.contents]}]
        except Exception as exc:
            return [{"error": f"MCP resource '{uri}' on server '{self.name}' failed: {exc}"}]

    @property
    def tools(self) -> list[CLIverTool]:
        return list(self._tools)

    async def close(self) -> None:
        """Close the session pool (when enabled), then drop discovered tools.

        Idempotent.  In-flight pooled calls should be drained first — closing
        a session is how the SDK cancels them.
        """
        if self._pool is not None:
            pool, self._pool = self._pool, None
            await pool.close()
        self._tools.clear()


class StdioAdapter(MCPServerAdapter):
    """MCP over stdio (subprocess)."""

    transport = "stdio"
    _required = ("command",)

    @asynccontextmanager
    async def _transport(self) -> AsyncIterator[StdioServerParameters]:
        config = self.config
        yield StdioServerParameters(
            command=str(config["command"]),
            args=[str(arg) for arg in config.get("args") or ()],
            env={str(key): str(value) for key, value in (config.get("env") or {}).items()} or None,
            cwd=config.get("cwd"),
        )


class StreamableHTTPAdapter(MCPServerAdapter):
    """MCP over Streamable HTTP.

    Headers ride a caller-owned ``httpx2.AsyncClient`` (the SDK transport has
    no ``headers`` parameter and only closes a client it created itself).
    """

    transport = "streamable-http"
    _required = ("url",)
    supports_session_pool = True

    @asynccontextmanager
    async def _transport(self) -> AsyncIterator[Any]:
        url = str(self.config["url"])
        headers = self.config.get("headers") or None
        if not headers:
            yield streamable_http_client(url, terminate_on_close=True)
            return
        async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30.0, read=300.0)) as http_client:
            yield streamable_http_client(url, http_client=http_client, terminate_on_close=True)


_TRANSPORTS: dict[str, type[MCPServerAdapter]] = {
    "stdio": StdioAdapter,
    "streamable-http": StreamableHTTPAdapter,
}


def create_adapter(name: str, config: dict[str, Any]) -> MCPServerAdapter:
    transport = config.get("transport", "stdio")
    cls = _TRANSPORTS.get(transport)
    if cls is None:
        raise ValueError(f"Unknown MCP transport '{transport}' for server '{name}'. Supported: {sorted(_TRANSPORTS)}")
    return cls(name=name, config=config)
