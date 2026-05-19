"""Pydantic models for CLIver labs and cells."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Cell input schemas — typed definitions for each cell type
# ──────────────────────────────────────────────────────────────────────────────


class ConfigFieldDef(BaseModel):
    """A single configuration field rendered by ConfigCell in the admin UI.

    Supported types:
      - ``text``       — plain text input
      - ``select``     — dropdown with predefined options
      - ``checkbox``   — boolean toggle
    """

    type: Literal["text", "select", "checkbox"] = "text"
    label: str = ""
    required: bool = False
    default: Any = None
    options: List[str] = Field(default_factory=list, description="Dropdown choices (select type only)")
    placeholder: str = ""


class LlmCellInputs(BaseModel):
    """Inputs for an LLM cell — prompt, agent selection, and verification."""

    prompt: str = ""
    agent: str = ""
    output_format: Literal["text", "json"] = "text"
    system_prompt: str = ""
    # Verification loop (optional)
    expected_result: str = ""
    max_retries: int = 1
    timeout_s: int = 120
    verifier_agent: str = ""


class CodeCellInputs(BaseModel):
    """Inputs for a Python code cell."""

    source: str = 'def run(ctx):\n    return {"result": "hello"}'
    hidden: bool = False


class DisplayCellInputs(BaseModel):
    """Inputs for a display/output cell."""

    content: str = ""
    format: Literal["markdown", "html"] = "markdown"


# ──────────────────────────────────────────────────────────────────────────────
# Cell
# ──────────────────────────────────────────────────────────────────────────────

CELL_STATUS = Literal["idle", "running", "completed", "error"]
CELL_TYPE = Literal["config", "llm", "code", "display"]


class Cell(BaseModel):
    """A single cell in a lab notebook.

    The ``inputs`` dict holds cell-type-specific data.  Use the typed
    input models above for creation and validation:

        config_cell.inputs = {"schema": {"topic": {"type": "text", "label": "Topic"}}}
        llm_cell.inputs = LlmCellInputs(prompt="Summarize").model_dump()
        code_cell.inputs = CodeCellInputs(source="print(42)").model_dump()
    """

    id: str
    type: str = "llm"  # "config" | "llm" | "code" | "display"
    title: str = ""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    status: CELL_STATUS = "idle"
    error: Optional[str] = None
    duration_ms: int = 0

    # ── Typed input accessors ──────────────────────────────────────────────

    def get_config_schema(self) -> Dict[str, ConfigFieldDef]:
        """Return parsed ConfigCell field definitions, or empty dict."""
        raw = self.inputs.get("schema", {})
        if isinstance(raw, dict):
            return {k: ConfigFieldDef(**v) if isinstance(v, dict) else ConfigFieldDef() for k, v in raw.items()}
        return {}

    def get_llm_inputs(self) -> LlmCellInputs:
        """Return typed LLM cell inputs."""
        return LlmCellInputs(**{k: v for k, v in self.inputs.items() if k in LlmCellInputs.model_fields})

    def get_code_inputs(self) -> CodeCellInputs:
        """Return typed code cell inputs."""
        return CodeCellInputs(**{k: v for k, v in self.inputs.items() if k in CodeCellInputs.model_fields})

    def get_display_inputs(self) -> DisplayCellInputs:
        """Return typed display cell inputs."""
        return DisplayCellInputs(**{k: v for k, v in self.inputs.items() if k in DisplayCellInputs.model_fields})


# ──────────────────────────────────────────────────────────────────────────────
# Lab
# ──────────────────────────────────────────────────────────────────────────────


class Lab(BaseModel):
    """A CLIver lab document."""

    schema_version: str = Field(default="cliver-lab-v1", alias="$schema")
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


class LabSummary(BaseModel):
    """Lightweight lab metadata for listing."""

    id: str
    title: str
    description: str = ""
    scenario_id: Optional[str] = None
    cell_count: int = 0
    status: str = "idle"
    created_at: str = ""
    updated_at: str = ""
