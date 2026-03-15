"""
Workflow models for CLIver Workflow Engine.

Simplified model hierarchy:
- Workflow: definition (steps, inputs)
- Step types: FunctionStep, LLMStep, HumanStep, WorkflowStep
- ExecutionContext: runtime state (inputs + step outputs)
- ExecutionResult: per-step result
- WorkflowExecutionState: full execution state (for persistence/resume)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Step types supported by the workflow engine."""

    FUNCTION = "function"
    LLM = "llm"
    WORKFLOW = "workflow"
    HUMAN = "human"


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------


class BaseStep(BaseModel):
    """Base step model — common fields for all step types."""

    id: str = Field(..., description="Unique identifier for the step")
    name: str = Field(..., description="Descriptive name of the step")
    type: StepType = Field(..., description="Type of the step")
    description: Optional[str] = Field(None, description="Description of the step")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Input variables for the step")
    outputs: Optional[List[str]] = Field(None, description="Output variable names from the step")
    skipped: bool = Field(False, description="Whether the step is skipped")


class FunctionStep(BaseStep):
    """Execute a Python function."""

    type: StepType = StepType.FUNCTION
    function: str = Field(..., description="Module path to the function (e.g. 'mymodule.func')")


class LLMStep(BaseStep):
    """Execute an LLM inference call."""

    type: StepType = StepType.LLM
    prompt: str = Field(..., description="Prompt for the LLM")
    model: Optional[str] = Field(None, description="LLM model to use")
    stream: bool = Field(False, description="Whether to stream the response")
    images: Optional[List[str]] = Field(None, description="Image files to send with the message")
    audio_files: Optional[List[str]] = Field(None, description="Audio files to send with the message")
    video_files: Optional[List[str]] = Field(None, description="Video files to send with the message")
    files: Optional[List[str]] = Field(None, description="General files to upload for tools")
    template: Optional[str] = Field(None, description="Template to use for the prompt")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for templates")


class WorkflowStep(BaseStep):
    """Execute a nested workflow."""

    type: StepType = StepType.WORKFLOW
    workflow: str = Field(..., description="Workflow name or path to execute")
    workflow_inputs: Optional[Dict[str, Any]] = Field(None, description="Inputs for the sub-workflow")


class HumanStep(BaseStep):
    """Pause for human input."""

    type: StepType = StepType.HUMAN
    prompt: str = Field(..., description="Prompt to show to the user")
    auto_confirm: bool = Field(False, description="Automatically confirm without user input")


# Union type for all step types
Step = Union[FunctionStep, LLMStep, WorkflowStep, HumanStep]


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------


class Workflow(BaseModel):
    """Workflow definition — a reusable template of steps."""

    name: str = Field(..., description="Unique name of the workflow")
    description: Optional[str] = Field(None, description="Description of the workflow")
    inputs: Optional[Dict[str, Any]] = Field(
        None, description="Input parameter defaults (name → default value)"
    )
    steps: List[Step] = Field(default_factory=list, description="Steps in the workflow")

    def get_initial_inputs(self, provided_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge provided inputs with defaults."""
        result = dict(self.inputs or {})
        if provided_inputs:
            result.update(provided_inputs)
        return result


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------


class ExecutionContext(BaseModel):
    """Runtime context for workflow execution.

    Holds workflow inputs and step outputs. Steps access previous
    step results via Jinja2 templating: {{ step_id.outputs.key }}
    """

    workflow_name: str = Field(..., description="Name of the workflow being executed")
    execution_id: Optional[str] = Field(None, description="Unique execution identifier")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Workflow input variables")
    steps: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Step results keyed by step ID → {outputs: {...}}",
    )


class ExecutionResult(BaseModel):
    """Result of a single step execution."""

    step_id: str = Field(..., description="ID of the executed step")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output variables")
    success: bool = Field(True, description="Whether the step succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class WorkflowExecutionState(BaseModel):
    """Full state of a workflow execution — persisted for resume."""

    workflow_name: str = Field(..., description="Name of the workflow")
    execution_id: str = Field(..., description="Unique execution identifier")
    current_step_index: int = Field(0, description="Index of the current step")
    completed_steps: List[str] = Field(default_factory=list, description="Completed step IDs")
    status: str = Field("running", description="running | paused | completed | failed | cancelled")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(None, description="Total execution time in seconds")
    context: ExecutionContext = Field(..., description="Execution context with all inputs and outputs")
