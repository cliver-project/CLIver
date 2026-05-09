---
name: workflow-create
description: Create a single workflow from a concrete step-by-step description. Best for small automations (under 10 steps) where the user already knows what steps are needed and wants a DAG with dependencies and conditional branching.
keywords: workflow, create, automate, pipeline, dag, steps
allowed-tools: Read LS Grep Write Ask Skill WorkflowValidate
---

# Create Workflow

You are creating a reusable CLIver workflow. Follow this process:

## 1. Understand the Goal

Ask the user what they want to automate. Clarify:
- What are the main steps?
- Are there conditions or branches?
- What inputs does the workflow need?
- Should any steps run in parallel?

## 2. Get the Schema (if needed)

If you are unsure about the exact fields, call:
```
WorkflowValidate(action='schema')
```
This returns the complete field reference. Skip for simple workflows.

## 3. Design the DAG

Plan the workflow as a directed acyclic graph:
- Each step has a unique `id` and a `type`: either `llm` or `python`
- **Step IDs MUST use underscores only** (e.g., `write_story`)
- Steps declare `depends_on` to specify execution order
- Use `condition` on steps for conditional branching (dot-path expressions)
- Steps listed in `depends_on` automatically inject their outputs as context
- Use `${ref}` for inline substitution: `${inputs.topic}`, `${step_id.result}`

## 4. Output Workflow YAML

Generate a valid YAML block. Key format rules:
- Only two step types: `llm` and `python`
- No `agents` section — `model`, `role`, `tools` are inline per step
- No Jinja2 — use `${ref}` for substitution, auto-inject from `depends_on`
- No `expected_result`, `retry`, `timeout`, `name`, `overview` fields

Example:

```yaml
name: my_workflow
description: What this workflow does

inputs:
  topic:
    type: string
    default: AI
    description: The research topic

steps:
  - id: research
    type: llm
    model: qwen
    role: "Research analyst"
    prompt: "Research ${inputs.topic} thoroughly"
    output_format: json

  - id: transform
    type: python
    file: ./scripts/transform.py
    depends_on: [research]

  - id: write_positive
    type: llm
    prompt: "Write a positive summary about ${inputs.topic}"
    depends_on: [research]
    condition: "research.sentiment == 'positive'"

  - id: write_negative
    type: llm
    prompt: "Write a critical analysis"
    depends_on: [research]
    condition: "research.sentiment == 'negative'"

  - id: summarize
    type: llm
    prompt: "Create a final summary"
    depends_on: [research]
    output_format: markdown
```

### Step fields

**LLM step** (`type: llm`):
- `prompt` (required): Prompt text, supports `${ref}` substitution
- `model` (optional): Override LLM model
- `role` (optional): Role description injected as system context
- `tools` (optional): Tool names to enable (allowlist)
- `output_format` (optional): `json` (default), `text`, or `markdown`
- `depends_on` (optional): Step IDs that must complete first
- `condition` (optional): Dot-path expression for conditional execution

**Python step** (`type: python`):
- `file` (required): Path to `.py` file relative to workflow directory
  - The file must define: `run(inputs: dict) -> dict`
- `depends_on` (optional): Step IDs that must complete first
- `condition` (optional): Dot-path expression for conditional execution

### Context injection
- Steps listed in `depends_on` automatically inject their outputs as context before the prompt
- Use `${ref}` for inline substitution: `${inputs.param}`, `${step_id.field}`, `${step_id.files[0].path}`

### Condition syntax
Conditions are safe dot-path expressions (not Jinja2, not Python eval):
- `research.sentiment == 'positive'`
- `step1.count > 5`
- `step1.success and step2.done`
- `not step1.error`

### Media files
LLM steps that generate files (images, audio, etc.) return them as file path references:
```json
{"result": "text...", "files": [{"type": "image", "path": "step_id/cover.png"}]}
```

## 5. Validate Before Saving

**You MUST validate every workflow YAML before saving.** Call:
```
WorkflowValidate(action='validate', yaml_content='...')
```
Fix any errors and re-validate until it returns `Valid`.
Never save a workflow that has not passed validation.

## 6. Save the Workflow

After validation passes, save to `.cliver/workflows/` using `Write`:
`.cliver/workflows/{name}.yaml`

Tell the user the workflow is saved and how to run it:
`/workflow run {name}`

## Rules
- Every step MUST have a unique `id` using underscores only
- Only two step types: `llm` and `python`
- No `agents` section — everything is inline per step
- Use `${ref}` not `{{ }}` for variable substitution
- Dependencies must reference existing step IDs (no cycles)
- Do NOT hardcode directory paths in prompts
- **Always validate, then save** — never save unvalidated YAML
