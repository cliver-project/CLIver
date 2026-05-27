"""Pydantic models for projects, issues, and scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class Project(BaseModel):
    """A project containing issues."""

    id: str
    name: str
    description: str = ""
    source: str = "local"
    source_url: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Issue(BaseModel):
    """An issue within a project."""

    id: str
    project_id: str
    title: str
    description: str = ""
    status: str = "open"
    priority: str = "medium"
    labels: List[str] = Field(default_factory=list)
    assigned_agent: Optional[str] = None
    scenario_id: Optional[str] = None
    lab_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


VALID_STATUSES = {"open", "in_progress", "completed", "closed"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}


class Scenario(BaseModel):
    """A scenario template for generating labs."""

    id: str
    name: str
    display_name: str
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    agent_requirements: List[str] = Field(default_factory=list)
    source: str = "builtin"
    path: Optional[str] = None
