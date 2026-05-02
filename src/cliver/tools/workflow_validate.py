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
    """Parse YAML through the Workflow Pydantic model, semantic checks, and LangGraph compilation."""
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

    # Phase 4: LangGraph compilation
    compile_errors = _compile_check(wf)
    if compile_errors:
        return "Compilation errors:\n" + "\n".join(f"- {e}" for e in compile_errors)

    step_summary = ", ".join(f"{s.id} ({s.type.value})" for s in wf.steps)
    return f"Valid. Workflow '{wf.name}' with {len(wf.steps)} steps: {step_summary}"


def _semantic_checks(wf) -> list[str]:
    """Check cross-step references, Jinja2 templates, and dependency graph."""
    import re

    from cliver.workflow.workflow_models import DecisionStep, LLMStep

    errors = []
    step_ids = {s.id for s in wf.steps}

    # --- Step ID format ---
    for s in wf.steps:
        if not re.match(r"^[a-z][a-z0-9_]*$", s.id):
            if "-" in s.id:
                errors.append(
                    f"Step id '{s.id}' contains hyphens. Use underscores instead "
                    f"(e.g., '{s.id.replace('-', '_')}') — hyphens break Jinja2 "
                    f"template references like {{{{ {s.id}.outputs.result }}}}"
                )
            else:
                errors.append(
                    f"Step id '{s.id}' is not a valid identifier. "
                    f"Use lowercase letters, digits, and underscores only (e.g., 'my_step_1')"
                )

    # --- Duplicate IDs ---
    seen = set()
    for s in wf.steps:
        if s.id in seen:
            errors.append(f"Duplicate step id: '{s.id}'")
        seen.add(s.id)

    # --- Dependency references ---
    for s in wf.steps:
        for dep in s.depends_on:
            if dep not in step_ids:
                errors.append(f"Step '{s.id}' depends_on unknown step '{dep}'")

        if isinstance(s, DecisionStep):
            for branch in s.branches:
                if branch.next_step not in step_ids:
                    errors.append(f"Decision '{s.id}' branch references unknown step '{branch.next_step}'")
            if s.default and s.default not in step_ids:
                errors.append(f"Decision '{s.id}' default references unknown step '{s.default}'")

    # --- Jinja2 template references (only if all step IDs are valid) ---
    has_invalid_ids = any(not re.match(r"^[a-z][a-z0-9_]*$", s.id) for s in wf.steps)
    if not has_invalid_ids:

        def check_templates(step, text, field_name):
            if not text or "{{" not in text:
                return
            refs = re.findall(r"\{\{\s*([a-zA-Z_][\w]*)", text)
            valid_vars = {"inputs", "steps"} | step_ids
            for ref in refs:
                if ref not in valid_vars:
                    errors.append(
                        f"Step '{step.id}' {field_name}: template reference '{ref}' "
                        f"is not a known step ID or variable. "
                        f"Available: inputs, steps, {', '.join(sorted(step_ids))}"
                    )

        for s in wf.steps:
            if hasattr(s, "prompt") and isinstance(getattr(s, "prompt", None), str):
                check_templates(s, s.prompt, "prompt")
            if s.condition:
                check_templates(s, s.condition, "condition")

    # --- Agent references ---
    agent_names = set(wf.agents.keys()) if wf.agents else set()
    for s in wf.steps:
        if isinstance(s, LLMStep) and s.agent:
            if s.agent not in agent_names:
                errors.append(
                    f"Step '{s.id}' references agent '{s.agent}' "
                    f"which is not defined in the agents section. "
                    f"Available agents: {', '.join(sorted(agent_names)) if agent_names else 'none'}"
                )

    # --- Cycle detection ---
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
    """Return concise field reference for all workflow step types."""
    return """# Workflow Model Reference

## Top-level fields
- name (str, required): Unique workflow name
- description (str, optional): What the workflow does
- overview (str, optional): High-level context shared with all subagents
- inputs (dict, optional): Default input parameters (key: default_value)
- agents (dict, optional): Reusable agent profile configs, keyed by agent name
  Example:
    agents:
      researcher:
        role: "Research analyst"
        instructions: "Focus on accuracy and citations."
- steps (list, required): List of step definitions
- permissions (optional): Permission overrides for the workflow

## Shared step fields (all step types)
- id (str, required): Unique step identifier — must be a valid Python identifier
  (lowercase letters, digits, underscores only, e.g. 'write_story'). Do NOT use
  hyphens — they break Jinja2 template references like {{ step_id.outputs.result }}
- name (str, required): Descriptive name
- type (str, required): "llm" | "function" | "human" | "decision" | "workflow"
- description (str, optional): Step description
- inputs (dict, optional): Input variables for the step
- outputs (list[str], optional): Named output variables
- depends_on (list[str], default=[]): Step IDs that must complete first
- condition (str, optional): Jinja2 expression — skip step if false
- expected_result (str, optional): Description of expected output — LLM validates the result
  and retries until expectations are met or timeout is reached
- retry (int, default=0): Max retries (0 = unlimited, keeps retrying until expected result or timeout)
- timeout (int, default=1800): Step timeout in seconds (default 30 minutes)

## LLM step (type: llm)
- prompt (str, required): Prompt for the LLM (supports Jinja2: {{ step_id.outputs.key }})
- model (str, optional): Override LLM model for this step
- agent (str, optional): Agent profile name from workflow agents section
- output_format (str, default='md'): Text output file format (md, json, txt, yaml, code)
- skills (list[str], optional): Skills to activate
- stream (bool, default=false): Whether to stream the response
- images (list[str], optional): Image file paths to include
- audio_files (list[str], optional): Audio file paths to include
- video_files (list[str], optional): Video file paths to include
- files (list[str], optional): General file paths to include
- permissions (optional): Permission overrides

## Step output references (available in Jinja2 templates)
- {{ step_id.outputs.result }} — text output of the step
- {{ step_id.outputs.media_files }} — list of generated media file paths
- {{ step_id.outputs.media_files[0] }} — first media file path
- {{ step_id.outputs.outputs_dir }} — output directory path

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
