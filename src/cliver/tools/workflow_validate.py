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
        "- **schema**: Returns the complete field reference for step types "
        "(llm, python) so you can write correct YAML.\n\n"
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
    """Parse YAML through the Workflow Pydantic model, semantic checks, and LangGraph compilation."""
    if not yaml_content or not yaml_content.strip():
        return "Error: No YAML content provided."

    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return f"YAML syntax error:\n{e}"

    if not isinstance(data, dict):
        return "Error: Workflow YAML must be a mapping (dict), not a list or scalar."

    from cliver.workflow.workflow_models import Workflow

    try:
        wf = Workflow(**data)
    except Exception as e:
        return f"Validation errors:\n{e}"

    errors = _semantic_checks(wf)
    if errors:
        return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)

    compile_errors = _compile_check(wf)
    if compile_errors:
        return "Compilation errors:\n" + "\n".join(f"- {e}" for e in compile_errors)

    step_summary = ", ".join(f"{s.id} ({s.type})" for s in wf.steps)
    return f"Valid. Workflow '{wf.name}' with {len(wf.steps)} steps: {step_summary}"


def _semantic_checks(wf) -> list[str]:
    """Check cross-step references and dependency graph."""
    import re

    errors = []
    step_ids = {s.id for s in wf.steps}

    for s in wf.steps:
        if not re.match(r"^[a-z][a-z0-9_]*$", s.id):
            if "-" in s.id:
                errors.append(
                    f"Step id '{s.id}' contains hyphens. Use underscores instead (e.g., '{s.id.replace('-', '_')}')"
                )
            else:
                errors.append(
                    f"Step id '{s.id}' is not a valid identifier. Use lowercase letters, digits, and underscores only"
                )

    seen = set()
    for s in wf.steps:
        if s.id in seen:
            errors.append(f"Duplicate step id: '{s.id}'")
        seen.add(s.id)

    for s in wf.steps:
        for dep in s.depends_on:
            if dep not in step_ids:
                errors.append(f"Step '{s.id}' depends_on unknown step '{dep}'")

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


def _compile_check(wf) -> list[str]:
    """Try to compile the workflow into a LangGraph StateGraph."""
    errors = []
    try:
        from cliver.workflow.compiler import WorkflowCompiler

        compiler = WorkflowCompiler()
        compiler.compile(wf)
    except Exception as e:
        errors.append(f"LangGraph compilation failed: {e}")
    return errors


def _format_schema() -> str:
    """Return concise field reference for workflow step types."""
    return """# Workflow Model Reference

## Top-level fields
- name (str, required): Unique workflow name
- description (str, optional): What the workflow does
- inputs (dict, optional): Input parameters with type/default/description
- steps (list, required): List of step definitions
- permissions (optional): Permission overrides (mode: yolo|auto-edit|default)

## Shared step fields (all step types)
- id (str, required): Unique step identifier (lowercase, underscores)
- type (str, required): "llm" | "python"
- depends_on (list[str], default=[]): Step IDs that must complete first
- condition (str, optional): Dot-path expression — skip step if false
  Examples: "research.sentiment == 'positive'", "s1.count > 5"

## LLM step (type: llm)
- prompt (str, required): Prompt text. Supports ${ref} substitution:
  ${inputs.topic}, ${step_id.result}, ${step_id.files[0].path}
- model (str, optional): Override LLM model for this step
- role (str, optional): Role description injected as system context
- tools (list[str], optional): Tool names to enable (allowlist)
- output_format (str, default='json'): json | text | markdown

## Python step (type: python)
- file (str, required): Path to .py file (relative to workflow dir)
  The file must define: run(inputs: dict) -> dict

## Context injection
- Steps listed in depends_on automatically inject their outputs as context
- Use ${ref} for inline substitution: ${inputs.param}, ${step_id.field}

## Example workflow
```yaml
name: research-pipeline
inputs:
  topic:
    type: string
    default: AI
steps:
  - id: research
    type: llm
    model: qwen
    role: "Research analyst"
    prompt: "Research ${inputs.topic}"
    output_format: json
  - id: summarize
    type: llm
    prompt: "Summarize the findings"
    depends_on: [research]
    output_format: markdown
```
"""


workflow_validate = WorkflowValidateTool()
