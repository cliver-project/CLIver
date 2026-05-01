---
name: wbs-planner
description: Decompose requirements into a Work Breakdown Structure and generate workflow YAML. Supports iterative updates when requirements change.
keywords: wbs, planning, decomposition, workflow, project, breakdown
allowed-tools: Read LS Grep Bash Write Ask Skill WorkflowValidate
---

# WBS Planner

Decompose user requirements into a hierarchical Work Breakdown Structure (WBS) and generate
executable workflow YAML files. Each actionable task becomes an LLM step. Complex work packages
become sub-workflows linked via `workflow_file`.

## 1. Analyze Requirements

Ask the user clarifying questions one at a time using `Ask`:
- What is the overall goal or deliverable?
- What are the major phases or milestones?
- Are there external dependencies, constraints, or deadlines?
- What is the expected scope — small (5-10 tasks), medium (10-20), or large (20+)?

If the user references an existing WBS or workflow, read it first with `Read` and
propose modifications rather than starting from scratch (see Step 7).

## 2. Decompose into WBS

Create a hierarchical breakdown with three levels:

- **Level 1: Phases** — major project phases (e.g., Design, Implementation, Testing, Deployment)
- **Level 2: Work Packages** — deliverable-scoped groups within each phase
- **Level 3: Tasks** — individual actionable items (each becomes an LLM step)

Present the WBS tree to the user as a numbered outline and confirm before proceeding:

```
1. Design Phase
   1.1 Requirements Analysis
       1.1.1 Gather stakeholder input
       1.1.2 Document functional requirements
   1.2 Architecture Design
       1.2.1 Design system components
       1.2.2 Define API contracts
2. Implementation Phase
   ...
```

## 3. Decide Workflow Structure

Based on the WBS complexity, choose a structure:

**Simple** (10 or fewer tasks, 1-2 phases):
- Single flat workflow YAML file
- All tasks as LLM steps in one file

**Complex** (more than 10 tasks, 3+ phases, or deeply nested work packages):
- A main orchestrator workflow with `type: workflow` steps
- Sub-workflow YAML files per work package, linked via `workflow_file`
- Sub-workflows stored in a `sub-workflows/` subdirectory

Present the proposed structure to the user via `Ask` and get confirmation.

## 4. Generate Workflow YAML

For each workflow file:

1. Each Level 3 task becomes a `type: llm` step with a specific, actionable prompt
2. Set `depends_on` based on task dependencies from the WBS
3. Add `type: decision` steps for conditional paths (if/else branching)
4. Add `type: human` steps for approval gates between phases
5. Use `overview` for shared project context visible to all steps
6. Use `agents` section for specialized roles (architect, developer, tester, etc.)
7. Use `inputs` for configurable parameters

For hierarchical structures, the main workflow references sub-workflows:
```yaml
steps:
  - id: design-phase
    type: workflow
    name: "Design Phase"
    workflow_file: ./sub-workflows/design-phase.yaml
    workflow_inputs:
      requirements: "{{ inputs.requirements }}"
```

Each sub-workflow file is self-contained with its own `name`, `overview`, `agents`, and `steps`.

Guidelines for LLM step prompts:
- Be specific and actionable — tell the LLM exactly what to produce
- Reference inputs and prior step outputs via Jinja2: `{{ inputs.param }}`, `{{ step_id.outputs.result }}`
- Set `output_format` appropriately (md for docs, json for structured data, code for source files)

## 5. Validate All Workflows

For each generated YAML, call:
```
WorkflowValidate(action='validate', yaml_content='...')
```
Fix any reported errors and re-validate until all pass.

## 6. Save Workflow Files

Save the main workflow:
```
Write(path='{workflows_dir}/{name}.yaml', content='...')
```

Save sub-workflows (if hierarchical):
```
Write(path='{workflows_dir}/sub-workflows/{package}.yaml', content='...')
```

Tell the user the workflow is saved and how to run it:
```
cliver workflow run {workflow-name}
```

## 7. Iterative Updates

When the user provides changed or additional requirements:

1. Read the existing workflow(s) with `Read`
2. Identify which WBS levels are affected by the change
3. Show a summary of proposed changes to the user:
   - Steps to add, remove, or modify
   - Dependencies to update
   - Sub-workflows affected
4. Get user confirmation via `Ask`
5. Modify only the affected workflows — preserve unchanged step IDs
6. Re-validate all modified workflows with `WorkflowValidate`
7. Save updated files with `Write`

Preserving step IDs is important — changing them invalidates existing checkpoint
state from prior workflow runs.

## Rules

- **Always validate before saving** — never save unvalidated YAML
- Step IDs: lowercase with hyphens, descriptive (e.g., `design-api`, `test-auth`)
- Every LLM step must have a clear, specific prompt — no vague instructions
- Max ~15 steps per workflow file — split into sub-workflows if larger
- Sub-workflow files go in a `sub-workflows/` subdirectory relative to the main workflow
- Use `{{ inputs.param }}` for configurable parameters
- Use `{{ step_id.outputs.result }}` for inter-step data flow
- When updating, preserve step IDs that haven't changed
- Each workflow file should have an `overview` providing context for its scope
- Use `type: human` steps at phase boundaries for user review gates
