---
title: Workflows
description: Define and execute sequential workflows using YAML
---

# Workflow Definition

CLIver has a workflow engine that executes a **linear sequence of steps** defined in YAML.
Each step runs in order, and its outputs are available to all subsequent steps.

Steps can be one of 4 types:

- `llm`:        LLM inference request
- `function`:   Python function call
- `human`:      Human interaction / confirmation
- `workflow`:   Nested sub-workflow

## Workflow Structure

A workflow YAML file has these top-level keys:

- `name`: Unique name for the workflow (required)
- `description`: Brief description (optional)
- `inputs`: Default input values as a dict (optional)
- `steps`: Ordered list of steps to execute (required)

## Basic Workflow Example

```yaml
name: research_analysis
description: Research a topic and provide analysis
inputs:
  topic: "How AI will change human's life"
steps:
  - id: research_topic
    name: Research Topic
    type: llm
    prompt: "Research the topic: {{ inputs.topic }}"
    model: deepseek-r1
    outputs: [research]

  - id: analyze_findings
    name: Analyze Findings
    type: llm
    prompt: "Analyze the research findings: {{ research_topic.outputs.research }}"
    model: qwen
    outputs: [analysis]

  - id: generate_report
    name: Generate Report
    type: llm
    prompt: "Generate a report based on: {{ analyze_findings.outputs.analysis }}"
    model: deepseek-r1
    outputs: [report]
```

## Step Properties

Each step has these properties:

- `id`: Unique identifier (required)
- `name`: Descriptive name (required)
- `type`: Step type — `llm`, `function`, `human`, or `workflow` (required)
- `description`: Description of what the step does (optional)
- `inputs`: Input variables for the step (optional)
- `outputs`: Output variable names (optional)
- `skipped`: Skip this step (optional, default: false)

## Variable Access

Steps reference data using Jinja2 templating:

| Pattern | Description |
|---------|-------------|
| `{{ inputs.key }}` | Workflow input parameter |
| `{{ key }}` | Shorthand for workflow input |
| `{{ step_id.outputs.key }}` | Output from a previous step |

### Example

```yaml
name: variable_access_example
description: Demonstrates variable access between steps
inputs:
  user_name: "User"
  topic: "AI"
steps:
  - id: greet
    name: Greet User
    type: function
    function: cliver.workflow.examples.compute_something
    inputs:
      greeting: "Hello {{ inputs.user_name }}!"
    outputs: [greeting, result]

  - id: analyze
    name: Analyze Topic
    type: function
    function: cliver.workflow.examples.process_results
    inputs:
      user_name: "{{ inputs.user_name }}"
      topic: "{{ inputs.topic }}"
      greeting: "{{ greet.outputs.greeting }}"
    outputs: [result]

  - id: summarize
    name: Create Summary
    type: llm
    prompt: |
      Summary for {{ inputs.user_name }} about {{ inputs.topic }}.
      Greeting: {{ greet.outputs.greeting }}
      Analysis: {{ analyze.outputs.result }}
    outputs: [summary]
```

## Step Types

### LLM Steps

```yaml
- id: analyze
  type: llm
  name: Analyze Topic
  prompt: "Analyze: {{ inputs.topic }}"
  model: qwen           # optional
  stream: false          # optional
  outputs: [analysis]
```

LLM steps also support: `images`, `audio_files`, `video_files`, `files`, `template`, `params`.

### Function Steps

```yaml
- id: compute
  type: function
  name: Process Data
  function: mypackage.utils.process_data
  inputs:
    data: "{{ previous_step.outputs.result }}"
  outputs: [result]
```

### Human Steps

```yaml
- id: confirm
  type: human
  name: User Confirmation
  prompt: "Continue with {{ analyze.outputs.result }}?"
  auto_confirm: false
```

### Workflow Steps

```yaml
- id: subflow
  type: workflow
  name: Run Analysis
  workflow: analysis_workflow
  workflow_inputs:
    topic: "{{ inputs.topic }}"
  outputs: [report]
```

## Workflow Inputs

Inputs are defined as a dict with optional defaults:

```yaml
inputs:
  topic: "default topic"      # has default value
  user_name: null              # required at runtime
```

Override at runtime:

```bash
cliver workflow run my_workflow -i topic="Custom topic" -i user_name="Alice"
```

## Running Workflows

```bash
# List available workflows
cliver workflow list

# Run a workflow
cliver workflow run workflow_name -i topic="AI"

# Resume a paused workflow
cliver workflow run workflow_name -e execution_id

# Dry run
cliver workflow run workflow_name --dry-run
```

## Workflow Directories

Workflows are loaded from:

1. `.cliver/workflows/` in the current directory (project-local, higher priority)
2. `~/.config/cliver/workflows/` (user-global)

Workflows are identified by their `name` field, not by file name.

## Persistence

Execution state is persisted to `~/.cache/cliver/` for pause/resume support:

```
~/.cache/cliver/{workflow_name}/{execution_id}/state.json
```

## Next Steps

Explore how to [extend CLIver](extensibility.md) or check the [roadmap](roadmap.md).
