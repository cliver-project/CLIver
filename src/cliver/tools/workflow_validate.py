"""Built-in tool for validating workflow YAML against the Workflow model.

The LLM calls this tool to check a workflow definition before saving it.
Two actions:
- validate: parse YAML through Pydantic and report errors
- schema:   return the complete field reference for all step types
"""

import logging
from typing import Literal, Type

import yaml
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkflowValidateInput(BaseModel):
    """Input schema for the WorkflowValidate tool."""

    action: Literal["validate", "schema"] = Field(
        description=(
            "'validate': parse the YAML and return errors or 'valid'. "
            "'schema': return the complete workflow model reference."
        )
    )
    yaml_content: str = Field(
        default="",
        description="The workflow YAML content to validate (required for 'validate' action).",
    )


class WorkflowValidateTool(BaseTool):
    """Validate workflow YAML or retrieve the workflow model schema."""

    name: str = "WorkflowValidate"
    description: str = (
        "Validate a workflow YAML definition before saving, or retrieve the "
        "complete workflow model schema.\n\n"
        "Actions:\n"
        "- **validate**: Pass workflow YAML content. Returns 'Valid' if the "
        "workflow parses correctly, or a list of specific errors to fix.\n"
        "- **schema**: Returns the complete field reference for all step types "
        "(LLM, function, human, decision, workflow) so you can write correct YAML.\n\n"
        "Always validate before saving a workflow with Write."
    )
    args_schema: Type[BaseModel] = WorkflowValidateInput
    tags: list = ["workflow", "validation"]

    def _run(self, action: str, yaml_content: str = "") -> str:
        if action == "schema":
            return _format_schema()
        if action == "validate":
            return _validate_yaml(yaml_content)
        return f"Unknown action: {action}. Use 'validate' or 'schema'."


def _validate_yaml(yaml_content: str) -> str:
    """Parse YAML through the Workflow Pydantic model and report errors."""
    if not yaml_content or not yaml_content.strip():
        return "Error: No YAML content provided."

    # Phase 1: YAML syntax
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return f"YAML syntax error:\n{e}"

    if not isinstance(data, dict):
        return "Error: Workflow YAML must be a mapping (dict), not a list or scalar."

    # Phase 2: Pydantic model validation
    from cliver.workflow.workflow_models import Workflow

    try:
        wf = Workflow(**data)
    except Exception as e:
        return f"Validation errors:\n{e}"

    # Phase 3: Semantic checks
    errors = _semantic_checks(wf)
    if errors:
        return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)

    step_summary = ", ".join(f"{s.id} ({s.type.value})" for s in wf.steps)
    return f"Valid. Workflow '{wf.name}' with {len(wf.steps)} steps: {step_summary}"


def _semantic_checks(wf) -> list[str]:
    """Check cross-step references and dependency graph."""
    from cliver.workflow.workflow_models import DecisionStep

    errors = []
    step_ids = {s.id for s in wf.steps}

    # Check for duplicate IDs
    seen = set()
    for s in wf.steps:
        if s.id in seen:
            errors.append(f"Duplicate step id: '{s.id}'")
        seen.add(s.id)

    for s in wf.steps:
        # Check depends_on references
        for dep in s.depends_on:
            if dep not in step_ids:
                errors.append(f"Step '{s.id}' depends_on unknown step '{dep}'")

        # Check decision branch references
        if isinstance(s, DecisionStep):
            for branch in s.branches:
                if branch.next_step not in step_ids:
                    errors.append(f"Decision '{s.id}' branch references unknown step '{branch.next_step}'")
            if s.default and s.default not in step_ids:
                errors.append(f"Decision '{s.id}' default references unknown step '{s.default}'")

    # Simple cycle detection (topological sort)
    if not errors:
        visited = set()
        in_progress = set()

        def has_cycle(sid):
            if sid in in_progress:
                return True
            if sid in visited:
                return False
            in_progress.add(sid)
            step = next((s for s in wf.steps if s.id == sid), None)
            if step:
                for dep in step.depends_on:
                    if has_cycle(dep):
                        return True
            in_progress.discard(sid)
            visited.add(sid)
            return False

        for s in wf.steps:
            if has_cycle(s.id):
                errors.append(f"Dependency cycle detected involving step '{s.id}'")
                break

    return errors


def _format_schema() -> str:
    """Return concise field reference for all workflow step types."""
    return """# Workflow Model Reference

## Top-level fields
- name (str, required): Unique workflow name
- description (str, optional): What the workflow does
- inputs (dict, optional): Default input parameters (key: default_value)
- steps (list, required): List of step definitions
- permissions (optional): Permission overrides for the workflow

## Shared step fields (all step types)
- id (str, required): Unique step identifier
- name (str, required): Descriptive name
- type (str, required): "llm" | "function" | "human" | "decision" | "workflow"
- description (str, optional): Step description
- inputs (dict, optional): Input variables for the step
- outputs (list[str], optional): Named output variables
- depends_on (list[str], default=[]): Step IDs that must complete first
- condition (str, optional): Jinja2 expression — skip step if false
- retry (int, default=0): Max retries on failure

## LLM step (type: llm)
- prompt (str, required): Prompt for the LLM (supports Jinja2: {{ step_id.outputs.key }})
- model (str, optional): Override LLM model for this step
- skills (list[str], optional): Skills to activate
- permissions (optional): Permission overrides

## Function step (type: function)
- function (str, required): Python module path (e.g., "mymodule.process")

## Human step (type: human)
- prompt (str, required): Prompt shown to the user
- auto_confirm (bool, default=false): Skip user input, auto-confirm

## Decision step (type: decision)
- branches (list, required): Evaluated in order, first match wins
  - condition (str, required): Jinja2 expression
  - next_step (str, required): Step ID to jump to
- default (str, optional): Fallback step ID if no branch matches

## Workflow step (type: workflow)
- workflow (str, optional): Name of another workflow to execute (from store)
- workflow_file (str, optional): Path to a workflow YAML file to execute
- workflow_inputs (dict, optional): Inputs to pass to the sub-workflow
NOTE: At least one must be set. If both are set, name is tried first, file is the fallback.

## Jinja2 template variables
- {{ inputs.param_name }} — workflow input parameters
- {{ step_id.outputs.result }} — LLM step output (auto-captured)
- {{ step_id.outputs.named_var }} — named output from a step
"""


workflow_validate = WorkflowValidateTool()
