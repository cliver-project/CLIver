"""Pydantic models for AI Lab and Golden Tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid4())[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Lab(BaseModel):
    """An AI Lab — an experimentation workspace for a domain agent solution."""

    id: str = Field(default_factory=_new_id)
    title: str
    description: str = ""
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class GoldenTest(BaseModel):
    """A golden test case for validating an AI Lab's agent behavior."""

    id: str = Field(default_factory=_new_id)
    lab_id: str
    name: str
    input: str
    expected_output: str
    expected_files: str = "[]"  # JSON array of expected file paths/descriptions
    sort_order: int = 0
