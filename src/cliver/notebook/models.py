"""Pydantic models for CLIver notebooks and cells."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Cell(BaseModel):
    """A single cell in a notebook."""

    id: str
    type: str  # "config" | "llm" | "code" | "display"
    title: str = ""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    status: str = "idle"  # "idle" | "running" | "completed" | "error"
    error: Optional[str] = None
    duration_ms: int = 0


class Notebook(BaseModel):
    """A CLIver notebook document."""

    schema_version: str = Field(default="cliver-notebook-v1", alias="$schema")
    id: str
    title: str
    description: str = ""
    scenario_id: Optional[str] = None
    default_agent: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cells: List[Cell] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    def model_dump(self, **kwargs):
        data = super().model_dump(by_alias=True, **kwargs)
        return data

    def get_cell(self, cell_id: str) -> Optional[Cell]:
        for c in self.cells:
            if c.id == cell_id:
                return c
        return None

    def cells_before(self, cell_id: str) -> List[Cell]:
        result = []
        for c in self.cells:
            if c.id == cell_id:
                break
            result.append(c)
        return result


class NotebookSummary(BaseModel):
    """Lightweight notebook metadata for listing."""

    id: str
    title: str
    description: str = ""
    scenario_id: Optional[str] = None
    cell_count: int = 0
    status: str = "idle"
    created_at: str = ""
    updated_at: str = ""
