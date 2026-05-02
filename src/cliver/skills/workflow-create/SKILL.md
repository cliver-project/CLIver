---
name: workflow-create
description: Create a single workflow from a concrete step-by-step description. Best for small automations (under 10 steps) where the user already knows what steps are needed and wants a DAG with dependencies, branching, and state propagation.
keywords: workflow, create, automate, pipeline, dag, steps
allowed-tools: Read LS Grep Write Ask Skill WorkflowValidate
---

# Create Workflow

You are creating a reusable CLIver workflow. Follow this process:

## 1. Understand the Goal

Ask the user what they want to automate. Clarify:
- What are the main steps?
- Are there conditions or branches (if X then do A, else do B)?
- What inputs does the workflow need?
- Should any steps run in parallel?

## 2. Get the Schema (if needed)

If you are unsure about the exact fields for a step type, call:
```
WorkflowValidate(action='schema')
```
This returns the complete field reference for all step types.
Only call this when you need it — skip for simple workflows.

## 3. Design the DAG

Plan the workflow as a directed acyclic graph:
- Each step has a unique `id` and a `type` (llm, function, human, decision, workflow)
- **Step IDs MUST use underscores only** (e.g., `write_story`). Hyphens break Jinja2.
- Steps declare `depends_on` to specify execution order
- Use `decision` type for if/else branching
- Steps pass results via Jinja2 templates: `{{ step_id.outputs.result }}`

## 4. Output Workflow YAML

Generate a valid YAML block. Key format rules:
- **agents** is a dict keyed by name, NOT a list
- **Step IDs** use underscores only, no hyphens
- Do NOT hardcode directory paths — the engine injects output directory automatically
- Use `expected_result` for steps where output quality matters

Example:

```yaml
name: my_workflow
description: What this workflow does
overview: |
  Context shared with all steps.

agents:
  writer:
    role: "Content writer"
    instructions: "Write clear content."

inputs:
  param_name: default_value

steps:
  - id: write_content
    type: llm
    name: Write Content
    prompt: "Write about {{ inputs.param_name }}"
    agent: writer
    output_format: md
    expected_result: "A complete draft with at least 3 paragraphs"
    timeout: 1800
    retry: 0

  - id: review_content
    type: llm
    name: Review
    prompt: "Review: {{ write_content.outputs.result }}"
    depends_on: [write_content]

  - id: branch_point
    type: decision
    name: Choose path
    depends_on: [review_content]
    branches:
      - condition: "'approved' in review_content.outputs.result"
        next_step: publish
      - condition: "'rejected' in review_content.outputs.result"
        next_step: revise
    default: revise
```

Output references for inter-step data flow:
- `{{ step_id.outputs.result }}` — text output
- `{{ step_id.outputs.media_files }}` — list of generated media file paths
- `{{ step_id.outputs.media_files[0] }}` — first media file
- `{{ inputs.param }}` — workflow input parameters

Validation fields:
- `expected_result`: LLM validates output meets this description, retries if not
- `retry`: 0 = unlimited retries until expected result or timeout
- `timeout`: seconds (default 1800 = 30 minutes)

## 5. Validate Before Saving

**You MUST validate every workflow YAML before saving.** Do NOT skip this step. Call:
```
WorkflowValidate(action='validate', yaml_content='...')
```
The validator checks YAML syntax, Pydantic model, step ID format (no hyphens),
dependency refs, template refs, agent refs, cycles, and LangGraph compilation.
Fix any errors and re-validate until it returns `Valid`.
Never call `Write` to save a workflow that has not passed validation.

## 6. Save the Workflow

After validation passes, save to the **project-local** `.cliver/workflows/` directory
(relative to CWD) using `Write`:
`.cliver/workflows/{name}.yaml`

Tell the user the workflow is saved and how to run it:
`cliver workflow run .cliver/workflows/{name}.yaml`

## Rules
- Every step MUST have a unique `id` using underscores only (no hyphens)
- Dependencies must reference existing step IDs (no cycles)
- Use `{{ step_id.outputs.result }}` for text, `{{ step_id.outputs.media_files[0] }}` for files
- Use `{{ inputs.param_name }}` to reference workflow inputs
- Do NOT hardcode directory paths in prompts
- Set `expected_result` on steps where output quality is critical
- agents is a dict keyed by name, NOT a list
- **Always validate, then save** — never save unvalidated YAML
