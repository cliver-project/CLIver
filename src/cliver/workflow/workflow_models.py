"""Workflow models — simplified for LangGraph-native execution."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class LLMStep(BaseModel):
    """A step that invokes an LLM with a prompt."""

    id: str
    type: Literal["llm"] = "llm"
    prompt: str
    model: Optional[str] = None
    role: Optional[str] = None
    tools: Optional[List[str]] = None
    output_format: Literal["json", "text", "markdown"] = "json"
    depends_on: List[str] = Field(default_factory=list)
    condition: Optional[str] = None


class PythonStep(BaseModel):
    """A step that runs a Python file's run(inputs) function."""

    id: str
    type: Literal["python"] = "python"
    file: str
    depends_on: List[str] = Field(default_factory=list)
    condition: Optional[str] = None


Step = Union[LLMStep, PythonStep]


def _parse_step(data: dict) -> Step:
    """Discriminate step type from the 'type' field."""
    step_type = data.get("type")
    if step_type == "llm":
        return LLMStep(**data)
    elif step_type == "python":
        return PythonStep(**data)
    raise ValueError(f"Unknown step type: {step_type!r}. Must be 'llm' or 'python'.")


class Workflow(BaseModel):
    """Top-level workflow definition."""

    name: str
    description: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    steps: List[Step]
    permissions: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def _parse_steps(cls, values):
        raw_steps = values.get("steps", [])
        parsed = []
        for s in raw_steps:
            if isinstance(s, dict):
                parsed.append(_parse_step(s))
            else:
                parsed.append(s)
        values["steps"] = parsed
        return values

    def get_initial_inputs(self, provided_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge provided inputs with defaults from the inputs schema."""
        result = {}
        if self.inputs:
            for key, schema in self.inputs.items():
                if isinstance(schema, dict) and "default" in schema:
                    result[key] = schema["default"]
        if provided_inputs:
            result.update(provided_inputs)
        return result
