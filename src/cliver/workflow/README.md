# CLIver Workflow Engine

A simple, sequential workflow engine for AI and automation tasks.

## Overview

The workflow engine executes a linear sequence of steps defined in YAML.
Each step runs in order, and its outputs are available to all subsequent steps
via Jinja2 templating. The engine supports pause/resume with persistent state.

## Features

- **YAML-based definitions**: Human-readable workflow files
- **4 step types**: LLM, Function, Workflow (nested), and Human
- **Result propagation**: Step outputs flow to subsequent steps via `{{ step_id.outputs.key }}`
- **Jinja2 templating**: Full Jinja2 support for variable resolution
- **Pause/Resume**: Execution state persisted to disk for continuation
- **Pure async**: No threading — clean async/await execution

## Quick Start

### Define a workflow

```yaml
name: research_analysis
description: Research a topic and generate a report
inputs:
  topic: "How AI will change human's life"
steps:
  - id: research
    name: Research Topic
    type: llm
    prompt: "Research the topic: {{ inputs.topic }}"
    outputs: [findings]

  - id: analyze
    name: Analyze Findings
    type: llm
    prompt: "Analyze: {{ research.outputs.findings }}"
    outputs: [analysis]

  - id: report
    name: Generate Report
    type: llm
    prompt: "Write a report based on: {{ analyze.outputs.analysis }}"
    outputs: [report]
```

### Run it

```bash
cliver workflow run research_analysis -i topic="Quantum computing"
```

## Variable Access

Steps access data using Jinja2 syntax:

| Pattern | Description |
|---------|-------------|
| `{{ inputs.key }}` | Workflow input parameter |
| `{{ key }}` | Shorthand for workflow input |
| `{{ step_id.outputs.key }}` | Output from a previous step |

### Example

```yaml
steps:
  - id: fetch
    type: function
    function: mymodule.fetch_data
    inputs:
      url: "{{ inputs.api_url }}"
    outputs: [data]

  - id: analyze
    type: llm
    prompt: "Analyze this data: {{ fetch.outputs.data }}"
    outputs: [analysis]

  - id: confirm
    type: human
    prompt: "Publish this analysis?\n{{ analyze.outputs.analysis }}"

  - id: publish
    type: function
    function: mymodule.publish
    inputs:
      content: "{{ analyze.outputs.analysis }}"
    outputs: [url]
```

## Step Types

### LLM Step

```yaml
- id: analyze
  type: llm
  name: Analyze Topic
  prompt: "Analyze: {{ inputs.topic }}"
  model: qwen          # optional, uses default model if omitted
  stream: false        # optional
  outputs: [analysis]
```

Also supports `images`, `audio_files`, `video_files`, `files`, `template`, `params`.

### Function Step

```yaml
- id: compute
  type: function
  name: Process Data
  function: mypackage.utils.process_data
  inputs:
    data: "{{ previous_step.outputs.result }}"
  outputs: [result]
```

### Human Step

```yaml
- id: confirm
  type: human
  name: User Confirmation
  prompt: "Continue with {{ analyze.outputs.result }}?"
  auto_confirm: false  # set true for automated pipelines
```

### Workflow Step (nested)

```yaml
- id: subflow
  type: workflow
  name: Run Sub-Analysis
  workflow: analysis_workflow
  workflow_inputs:
    topic: "{{ inputs.topic }}"
  outputs: [report]
```

## Workflow Inputs

Inputs are defined as a dict with default values:

```yaml
inputs:
  topic: "default topic"      # has default
  user_name: null              # required, no default
```

Override at runtime:

```bash
cliver workflow run my_workflow -i topic="Custom topic" -i user_name="Alice"
```

## Persistence

Execution state is saved to `~/.cache/cliver/` as JSON:

```
~/.cache/cliver/{workflow_name}/{execution_id}/state.json
```

This enables pause/resume:

```bash
# Pause a running workflow
cliver workflow pause workflow_name -e execution_id

# Resume it later
cliver workflow run workflow_name -e execution_id
```

## Workflow Directories

Workflows are discovered from:

1. `.cliver/workflows/` in the current directory (project-local)
2. `~/.config/cliver/workflows/` (user-global)

Project-local workflows take precedence over global ones with the same name.

## API Usage

```python
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager

manager = LocalDirectoryWorkflowManager()
executor = WorkflowExecutor(task_executor=my_task_executor, workflow_manager=manager)

result = await executor.execute_workflow("my_workflow", inputs={"topic": "AI"})
print(result.status)  # "completed"
print(result.context.steps["analyze"]["outputs"]["analysis"])
```
