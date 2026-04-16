"""
Workflow models for CLIver Workflow Engine.

Model hierarchy:
- Workflow: definition (steps, inputs, permissions)
- Step types: LLMStep, FunctionStep, HumanStep, WorkflowStep, DecisionStep
- ExecutionContext: runtime state (inputs + step outputs)
- ExecutionResult: per-step result
- WorkflowExecutionState: full execution state (for persistence/resume)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class StepType(str, Enum):
    FUNCTION = "function"
    LLM = "llm"
    WORKFLOW = "workflow"
    HUMAN = "human"
    DECISION = "decision"


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------


class BaseStep(BaseModel):
    id: str = Field(..., description="Unique identifier for the step")
    name: str = Field(..., description="Descriptive name of the step")
    type: StepType = Field(..., description="Type of the step")
    description: Optional[str] = Field(None, description="Description of the step")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Input variables for the step")
    outputs: Optional[List[str]] = Field(None, description="Output variable names from the step")
    skipped: bool = Field(False, description="Whether the step is skipped")
    depends_on: List[str] = Field(default_factory=list, description="Step IDs that must complete first")
    condition: Optional[str] = Field(None, description="Jinja2 expression — skip step if false")
    retry: int = Field(0, description="Max retries on failure")


class LLMStep(BaseStep):
    type: StepType = StepType.LLM
    prompt: str = Field(..., description="Prompt for the LLM")
    model: Optional[str] = Field(None, description="LLM model to use")
    stream: bool = Field(False, description="Whether to stream the response")
    skills: Optional[List[str]] = Field(None, description="Skills to activate for this step")
    images: Optional[List[str]] = Field(None, description="Image files")
    audio_files: Optional[List[str]] = Field(None, description="Audio files")
    video_files: Optional[List[str]] = Field(None, description="Video files")
    files: Optional[List[str]] = Field(None, description="General files")
    template: Optional[str] = Field(None, description="Template name")
    params: Optional[Dict[str, Any]] = Field(None, description="Template parameters")
    permissions: Optional[Any] = Field(None, description="Permission overrides")


class FunctionStep(BaseStep):
    type: StepType = StepType.FUNCTION
    function: str = Field(..., description="Module path to the function")


class HumanStep(BaseStep):
    type: StepType = StepType.HUMAN
    prompt: str = Field(..., description="Prompt to show to the user")
    auto_confirm: bool = Field(False, description="Auto-confirm without input")


class WorkflowStep(BaseStep):
    type: StepType = StepType.WORKFLOW
    workflow: str = Field(..., description="Workflow name to execute")
    workflow_inputs: Optional[Dict[str, Any]] = Field(None, description="Inputs for sub-workflow")


class Branch(BaseModel):
    condition: str = Field(..., description="Jinja2 condition expression")
    next_step: str = Field(..., description="Step ID to jump to if condition is true")


class DecisionStep(BaseStep):
    type: StepType = StepType.DECISION
    branches: List[Branch] = Field(..., description="Branches evaluated in order")
    default: Optional[str] = Field(None, description="Fallback step ID if no branch matches")


Step = Union[LLMStep, FunctionStep, HumanStep, WorkflowStep, DecisionStep]


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------


class Workflow(BaseModel):
    name: str = Field(..., description="Unique name of the workflow")
    description: Optional[str] = Field(None, description="Description")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Input parameter defaults")
    steps: List[Step] = Field(default_factory=list, description="Steps in the workflow")
    permissions: Optional[Any] = Field(None, description="Permission overrides")

    def get_initial_inputs(self, provided_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = dict(self.inputs or {})
        if provided_inputs:
            result.update(provided_inputs)
        return result


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------


class ExecutionContext(BaseModel):
    workflow_name: str = Field(..., description="Name of the workflow being executed")
    execution_id: Optional[str] = Field(None, description="Unique execution identifier")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Workflow input variables")
    steps: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Step results keyed by step ID",
    )


class ExecutionResult(BaseModel):
    step_id: str = Field(..., description="ID of the executed step")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output variables")
    success: bool = Field(True, description="Whether the step succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class WorkflowExecutionState(BaseModel):
    workflow_name: str
    execution_id: str
    completed_steps: List[str] = Field(default_factory=list)
    skipped_steps: List[str] = Field(default_factory=list)
    status: str = Field(
        "running",
        description="running | paused | completed | failed | cancelled",
    )
    error: Optional[str] = None
    execution_time: Optional[float] = None
    context: ExecutionContext
