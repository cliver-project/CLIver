"""Pydantic models for MCP Server."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid4())[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MCPServer(BaseModel):
    """An MCP server configuration stored in SQLite."""

    id: str = Field(default_factory=_new_id)
    name: str
    transport: str = "stdio"
    url: Optional[str] = None
    auth: Optional[str] = None  # JSON: {"type":"api_key","token":"..."} or {"type":"token","token":"..."}
    headers: Optional[str] = None  # JSON object string
    command: Optional[str] = None
    args: Optional[str] = None  # JSON array string
    envs: Optional[str] = None  # JSON object string
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
