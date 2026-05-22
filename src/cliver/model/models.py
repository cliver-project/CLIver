"""Pydantic models for Provider, Endpoint, and Model stored in SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid4())[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Provider(BaseModel):
    """An LLM provider stored in SQLite."""

    id: str = Field(default_factory=_new_id)
    name: str
    type: str = "openai"
    api_key: Optional[str] = None
    rate_limit: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class Endpoint(BaseModel):
    """A provider base URL endpoint."""

    id: str = Field(default_factory=_new_id)
    provider_id: str
    base_url: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class Model(BaseModel):
    """An LLM model configuration stored in SQLite."""

    id: str = Field(default_factory=_new_id)
    provider_id: str
    endpoint_id: str
    name: str
    capabilities: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)
    think_mode: Optional[int] = None
    context_window: Optional[int] = None
    pricing: Optional[Dict[str, Any]] = None
    is_default: int = 0
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
