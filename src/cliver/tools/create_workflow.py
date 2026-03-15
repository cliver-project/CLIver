"""Built-in create_workflow tool for dynamic workflow generation.

The LLM generates a workflow YAML definition for complex tasks (5+ steps)
that benefit from separate execution contexts per step. The tool validates
the YAML, saves it to .cliver/workflows/, and returns instructions to run it.
"""

import logging
from pathlib import Path
from typing import Optional, Type

import yaml
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.workflow.workflow_models import Workflow

logger = logging.getLogger(__name__)


class CreateWorkflowInput(BaseModel):
    """Input schema for create_workflow."""

    workflow_yaml: str = Field(
        description=(
            "The complete workflow definition in YAML format. "
            "Must include 'name' and 'steps' fields. "
            "Each step needs 'id', 'name', 'type', and type-specific fields."
        )
    )
    execute: Optional[bool] = Field(
        default=False,
        description="If true, also print the command to execute the workflow.",
    )


class CreateWorkflowTool(BaseTool):
    """Generate and save a workflow definition for complex multi-step tasks."""

    name: str = "create_workflow"
    description: str = (
        "Generate a workflow definition for complex tasks that need 5+ steps, "
        "different models per step, or long-running operations that benefit from "
        "pause/resume. The workflow is saved and can be executed separately.\n\n"
        "The workflow YAML format:\n"
        "```yaml\n"
        "name: workflow-name\n"
        "description: What it does\n"
        "inputs:\n"
        "  param_name: default_value\n"
        "steps:\n"
        "  - id: step_id\n"
        "    name: Step Name\n"
        "    type: llm          # llm, function, human, or workflow\n"
        "    prompt: \"...\"\n"
        "    model: model_name  # optional\n"
        "    outputs: [result]\n"
        "  - id: next_step\n"
        "    name: Next Step\n"
        "    type: llm\n"
        "    prompt: \"Use previous: {{ step_id.outputs.result }}\"\n"
        "    outputs: [final]\n"
        "```\n\n"
        "Guidelines:\n"
        "- Each step should do ONE thing well\n"
        "- 3-8 steps is ideal\n"
        "- Reference previous step outputs with {{ step_id.outputs.key }}\n"
        "- Reference workflow inputs with {{ inputs.key }}\n"
        "- Keep the workflow linear — no complex branching\n"
        "- Specify different models per step when beneficial"
    )
    args_schema: Type[BaseModel] = CreateWorkflowInput
    tags: list = ["planning", "workflow"]

    def _run(self, workflow_yaml: str, execute: bool = False) -> str:
        # Parse the YAML
        try:
            data = yaml.safe_load(workflow_yaml)
        except yaml.YAMLError as e:
            return f"Error: Invalid YAML syntax: {e}"

        if not isinstance(data, dict):
            return "Error: Workflow YAML must be a mapping (dict), not a list or scalar."

        # Validate as a Workflow model
        try:
            workflow = Workflow(**data)
        except Exception as e:
            return f"Error: Invalid workflow definition: {e}"

        if not workflow.steps:
            return "Error: Workflow must have at least one step."

        # Save to .cliver/workflows/
        workflows_dir = Path.cwd() / ".cliver" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{workflow.name}.yaml"
        filepath = workflows_dir / filename
        filepath.write_text(workflow_yaml, encoding="utf-8")

        logger.info(f"Workflow '{workflow.name}' saved to {filepath}")

        # Build response
        step_summary = "\n".join(
            f"  {i + 1}. [{s.type.value}] {s.name}" for i, s in enumerate(workflow.steps)
        )

        result = (
            f"Workflow '{workflow.name}' created with {len(workflow.steps)} steps:\n"
            f"{step_summary}\n\n"
            f"Saved to: {filepath}"
        )

        if execute:
            inputs_str = ""
            if workflow.inputs:
                inputs_str = " ".join(f'-i {k}="{v}"' for k, v in workflow.inputs.items() if v is not None)
            result += f"\n\nTo run: cliver workflow run {workflow.name}"
            if inputs_str:
                result += f" {inputs_str}"

        return result


create_workflow = CreateWorkflowTool()
