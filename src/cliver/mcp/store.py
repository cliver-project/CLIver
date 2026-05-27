"""MCPServerStore — SQLite persistence for MCP server configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.db import SQLiteStore
from cliver.mcp.models import MCPServer

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    transport TEXT NOT NULL DEFAULT 'stdio',
    url TEXT,
    auth TEXT,
    headers TEXT,
    command TEXT,
    args TEXT,
    envs TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class MCPServerStore:
    """CRUD store for MCP server configurations.

    Instantiate with a db_path, or use ``MCPServerStore.from_config_dir(config_dir)``
    to resolve the default ``cliver.db`` path automatically.
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._store: Optional[SQLiteStore] = None

    @classmethod
    def from_config_dir(cls, config_dir: Path) -> "MCPServerStore":
        """Create a store using the default database path for a config directory."""
        return cls(config_dir / "cliver.db")

    def _get_store(self) -> SQLiteStore:
        if self._store is None:
            self._store = SQLiteStore(self._db_path)
            self._store.execute_schema(_SCHEMA)
        return self._store

    # -- CRUD ----------------------------------------------------------------

    def list_servers(self) -> List[MCPServer]:
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, name, transport, url, auth, headers, command, args, envs, created_at, updated_at "
                "FROM mcp_servers ORDER BY updated_at DESC"
            ).fetchall()
        return [MCPServer(**dict(r)) for r in rows]

    def get_server(self, server_id: str) -> Optional[MCPServer]:
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, name, transport, url, auth, headers, command, args, envs, created_at, updated_at "
                "FROM mcp_servers WHERE id = ?",
                (server_id,),
            ).fetchone()
        if row is None:
            return None
        return MCPServer(**dict(row))

    def create_server(
        self,
        name: str,
        transport: str = "stdio",
        url: Optional[str] = None,
        auth: Optional[str] = None,
        headers: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[str] = None,
        envs: Optional[str] = None,
    ) -> MCPServer:
        server = MCPServer(
            name=name,
            transport=transport,
            url=url,
            auth=auth,
            headers=headers,
            command=command,
            args=args,
            envs=envs,
        )
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO mcp_servers (id, name, transport, url, auth, headers, command, args, envs, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    server.id,
                    server.name,
                    server.transport,
                    server.url,
                    server.auth,
                    server.headers,
                    server.command,
                    server.args,
                    server.envs,
                    server.created_at,
                    server.updated_at,
                ),
            )
        return server

    def update_server(
        self,
        server_id: str,
        name: Optional[str] = None,
        transport: Optional[str] = None,
        url: Optional[str] = None,
        auth: Optional[str] = None,
        headers: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[str] = None,
        envs: Optional[str] = None,
    ) -> Optional[MCPServer]:
        existing = self.get_server(server_id)
        if existing is None:
            return None
        if name is not None:
            existing.name = name
        if transport is not None:
            existing.transport = transport
        if url is not None:
            existing.url = url
        if auth is not None:
            existing.auth = auth
        if headers is not None:
            existing.headers = headers
        if command is not None:
            existing.command = command
        if args is not None:
            existing.args = args
        if envs is not None:
            existing.envs = envs

        from cliver.mcp.models import _now

        existing.updated_at = _now()
        with self._get_store().write() as db:
            db.execute(
                "UPDATE mcp_servers SET name=?, transport=?, url=?, auth=?, headers=?, command=?, args=?, envs=?, updated_at=? "
                "WHERE id=?",
                (
                    existing.name,
                    existing.transport,
                    existing.url,
                    existing.auth,
                    existing.headers,
                    existing.command,
                    existing.args,
                    existing.envs,
                    existing.updated_at,
                    server_id,
                ),
            )
        return existing

    def delete_server(self, server_id: str) -> bool:
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
        return cursor.rowcount > 0

    def close(self) -> None:
        if self._store:
            self._store.close()
            self._store = None

    # -- Conversion helpers ---------------------------------------------------

    def to_connection_dict(self, server: MCPServer) -> Dict[str, Any]:
        """Convert an MCPServer to a dict suitable for MCPServersCaller."""
        result: Dict[str, Any] = {"transport": server.transport}

        if server.transport == "stdio":
            if server.command:
                result["command"] = server.command
            if server.args:
                try:
                    result["args"] = json.loads(server.args)
                except (json.JSONDecodeError, TypeError):
                    pass
            if server.envs:
                try:
                    result["env"] = json.loads(server.envs)
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            # sse, streamable_http, websocket
            if server.url:
                result["url"] = server.url
            if server.headers:
                try:
                    result["headers"] = json.loads(server.headers)
                except (json.JSONDecodeError, TypeError):
                    pass
            if server.auth:
                try:
                    auth_data = json.loads(server.auth)
                    token = auth_data.get("token", "")
                    if token:
                        result["headers"] = result.get("headers") or {}
                        if auth_data.get("type") == "api_key":
                            result["headers"]["x-api-key"] = token
                        else:
                            # Default to Bearer token
                            result["headers"]["Authorization"] = f"Bearer {token}"
                except (json.JSONDecodeError, TypeError):
                    pass

        return result

    def get_connection_dicts(self) -> Dict[str, Dict[str, Any]]:
        """Get all servers as a {name: connection_dict} mapping for MCPServersCaller."""
        servers = self.list_servers()
        return {s.name: self.to_connection_dict(s) for s in servers}
