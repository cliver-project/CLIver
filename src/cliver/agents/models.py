"""Pydantic model for Agent stored in SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid4())[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Agent(BaseModel):
    """An agent configuration stored in SQLite."""

    id: str = Field(default_factory=_new_id)
    name: str
    type: str = "cliver"
    description: Optional[str] = None
    role: Optional[str] = None
    model: Optional[str] = None
    is_default: int = 0
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
