---
name: workflow-create
description: Create a reusable workflow from a natural language description. The LLM designs a DAG of steps with dependencies, branching, and state propagation, then outputs valid workflow YAML.
keywords: workflow, create, automate, pipeline, dag, steps
allowed-tools: Read LS Grep Bash Write Ask Skill
---

# Create Workflow

You are creating a reusable CLIver workflow. Follow this process:

## 1. Understand the Goal

Ask the user what they want to automate. Clarify:
- What are the main steps?
- Are there conditions or branches (if X then do A, else do B)?
- What inputs does the workflow need?
- Should any steps run in parallel?

## 2. Design the DAG

Plan the workflow as a directed acyclic graph:
- Each step has a unique `id` and a `type` (llm, function, human, decision, workflow)
- Steps declare `depends_on` to specify execution order
- Use `DecisionStep` for if/else branching
- Steps pass results via `outputs` and Jinja2 templates: `{{ step_id.outputs.key }}`

## 3. Output Workflow YAML

Generate a valid YAML block matching this schema:

```yaml
name: workflow-name
description: What this workflow does
inputs:
  param_name: default_value

steps:
  - id: unique_id
    type: llm                           # llm | function | human | decision | workflow
    name: Human-readable name
    prompt: "Your prompt here"          # For LLM steps
    outputs: [named_output]             # Optional: name the output for downstream steps
    depends_on: [other_step_id]         # Optional: dependencies
    condition: "jinja2 expression"      # Optional: skip if false
    retry: 0                            # Optional: retry count on failure

  - id: branch_point
    type: decision
    name: Choose path
    depends_on: [previous_step]
    branches:
      - condition: "'success' in previous_step.outputs.result"
        next_step: success_step
      - condition: "'failure' in previous_step.outputs.result"
        next_step: failure_step
    default: failure_step               # Fallback if no branch matches

  - id: human_approval
    type: human
    name: Get approval
    prompt: "Approve deployment to production?"
    depends_on: [build_step]
```

## 4. Save the Workflow

After generating the YAML, save it using `Write` to:
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
