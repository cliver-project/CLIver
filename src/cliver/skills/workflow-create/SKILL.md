---
name: workflow-create
description: Create a reusable workflow from a natural language description. The LLM designs a DAG of steps with dependencies, branching, and state propagation, then outputs valid workflow YAML.
keywords: workflow, create, automate, pipeline, dag, steps
allowed-tools: Read LS Grep Bash Write Ask Skill WorkflowValidate
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
- Steps declare `depends_on` to specify execution order
- Use `decision` type for if/else branching
- Steps pass results via `outputs` and Jinja2 templates: `{{ step_id.outputs.key }}`

## 4. Output Workflow YAML

Generate a valid YAML block. Example:

```yaml
name: workflow-name
description: What this workflow does
inputs:
  param_name: default_value

steps:
  - id: unique_id
    type: llm
    name: Human-readable name
    prompt: "Your prompt here"
    outputs: [named_output]
    depends_on: [other_step_id]

  - id: branch_point
    type: decision
    name: Choose path
    depends_on: [unique_id]
    branches:
      - condition: "'success' in unique_id.outputs.result"
        next_step: success_step
      - condition: "'failure' in unique_id.outputs.result"
        next_step: failure_step
    default: failure_step

  - id: human_approval
    type: human
    name: Get approval
    prompt: "Approve deployment to production?"
    depends_on: [branch_point]
```

## 5. Validate Before Saving

**Always validate before saving.** Call:
```
WorkflowValidate(action='validate', yaml_content='...')
```
This checks YAML syntax, required fields, step type schemas,
dependency references, and cycle detection. Fix any reported
errors and re-validate until it returns `Valid`.

## 6. Save the Workflow

After validation passes, save using `Write` to:
`{agent_workflows_dir}/{workflow-name}.yaml`

Tell the user the workflow is saved and how to run it:
`cliver workflow run {workflow-name}`

## Rules
- Every step MUST have a unique `id`
- Step IDs should be short, descriptive, lowercase with hyphens
- Dependencies must reference existing step IDs (no cycles)
- LLM steps always capture output as `outputs.result`
- Use `{{ step_id.outputs.result }}` to reference previous step results
- Use `{{ inputs.param_name }}` to reference workflow inputs
- DecisionStep branches are evaluated in order — first match wins
- Keep workflows focused — one workflow per automation goal
- **Always validate, then save** — never save unvalidated YAML
