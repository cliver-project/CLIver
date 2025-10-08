# Cliver Workflow Engine

A modular, extensible workflow engine for AI and automation tasks in CLIver.

## Overview

The Cliver Workflow Engine provides a deterministic, local-only workflow execution system.
It enables users to define complex automation workflows using YAML and execute them with built-in support for:
- LLM interactions
- function calls
- sub-workflows
- human confirmation steps.

## Features

- **YAML-based Workflow Definitions**: Define workflows in human-readable YAML format
- **Multiple Step Types**: LLM, Function, Workflow, and Human steps
- **Variable Resolution**: Reference inputs and outputs using `{{ variable }}` syntax with Jinja2 templating
- **Retry Logic**: Configurable retry policies with exponential backoff
- **Error Handling**: Flexible error handling with continue/fail/retry options
- **Conditional Execution**: Execute steps based on conditions
- **Pause/Resume**: Persistent execution state for workflow continuation
- **Extensible Design**: Plugin architecture for custom step types and persistence

## Getting Started

### Workflow Definition

Create a workflow in YAML format:

```yaml
name: example_workflow
description: A simple example workflow
version: "1.0"
inputs:
  - user_name
  - topic
steps:
  - id: greet
    type: llm
    name: Generate Greeting
    prompt: "Write a friendly greeting for {{ inputs.user_name }} about {{ inputs.topic }}"
    outputs: [greeting]

  - id: confirm
    type: human
    name: Confirm Action
    prompt: "Continue with greeting: {{ greet.greeting }}?"
    auto_confirm: false

  - id: process
    type: function
    name: Process Greeting
    function: mypackage.my_module.my_function
    inputs:
      text: {{ greet.greeting }}
    outputs: [result]
```

### Running Workflows

```bash
# List available workflows
cliver workflow list

# Run a workflow
cliver workflow run examples/hello_world.yaml --input user_name "Alice" --input topic "AI"

# Dry-run to validate
cliver workflow run examples/hello_world.yaml --input user_name "Alice" --dry-run

# View execution status
cliver workflow status
```

## Step Types

### LLM Step

Execute LLM inference with full Cliver integration:

```yaml
- id: analyze
  type: llm
  name: Analyze Topic
  prompt: "Analyze the topic: {{ inputs.topic }}"
  model: gpt-4
  stream: false
  skill_sets: [research, analysis]
  outputs: [analysis]
  retry:
    max_attempts: 3
    backoff_factor: 1.5
```

LLM steps also support multimedia content:

```yaml
- id: analyze_image
  type: llm
  name: Image Analysis
  prompt: "Analyze the image and describe what you see."
  model: gpt-4-vision
  images:
    - {{ inputs.image_path }}
  outputs: [description]
```

### Function Step

Execute Python functions dynamically:

```yaml
- id: compute
  type: function
  name: Process Data
  function: mypackage.utils.process_data
  inputs:
    data: {{ previous_step.output }}
  outputs: [result]
```

### Workflow Step

Execute sub-workflows:

```yaml
- id: subflow
  type: workflow
  name: Run Analysis
  workflow: analysis_workflow
  workflow_inputs:
    topic: {{ inputs.topic }}
  outputs: [report]
```

### Human Step

Wait for user confirmation:

```yaml
- id: confirm
  type: human
  name: User Confirmation
  prompt: "Continue with {{ previous_step.result }}?"
  auto_confirm: false
```

## Advanced Features

### Retry Policies

Configure retry behavior for steps:

```yaml
retry:
  max_attempts: 5
  backoff_factor: 2.0
  max_backoff: 60.0
```

### Error Handling

Define error handling behavior:

```yaml
on_error: continue  # or "fail" or "retry"
```

### Conditional Execution

Execute steps based on conditions:

```yaml
condition: "'success' in {{ previous_step.status }}"
```

### Variable Resolution

Reference variables using `{{ }}` syntax with Jinja2 templating:

- `{{ inputs.variable_name }}` - Workflow inputs
- `{{ step_id.output_name }}` - Step outputs
- `{{ variable_name }}` - Context variables

## Persistence

Workflow execution state is persisted locally by default in `~/.cache/cliver/`. This enables:

- Pause/resume capabilities
- Execution history tracking
- Recovery from interruptions

### Cache Structure

The persistence provider saves both the complete workflow execution state and individual step results:

- Workflow execution states: `{cache_dir}/{workflow_name}/{execution_id}/state.json`
- Step execution results: `{cache_dir}/{workflow_name}/{execution_id}/{step_id}_result.json`
- Multimedia content: `{cache_dir}/{workflow_name}/{execution_id}/{step_id}/media/{media_type}/`

This structure allows for efficient caching and retrieval of both the overall workflow state and individual step results.

## Extensibility

The workflow engine is designed to be extensible:

1. **Custom Step Types**: Implement new step types by extending `StepExecutor`
2. **Persistence Providers**: Add remote storage options
3. **Step Filters**: Modify step behavior through plugins

## API Reference

### Workflow Models

Pydantic models for type-safe workflow definitions:

- `Workflow`: Main workflow definition
- `BaseStep`: Base class for all step types
- `LLMStep`, `FunctionStep`, `WorkflowStep`, `HumanStep`: Specific step types
- `ExecutionContext`: Runtime execution context
- `ExecutionResult`: Step execution results (cached individually)
- `WorkflowExecutionState`: Complete workflow execution state (cached as a whole)

## Examples

See the `examples/` directory for complete workflow examples:

- `hello_world.yaml`: Basic workflow demonstration
- `simple_workflow.yaml`: Simple analysis workflow
- `advanced_workflow.yaml`: Advanced features demonstration
- `full_demo_workflow.yaml`: Complete feature showcase
- `multimedia_workflow.yaml`: Multimedia capabilities demonstration

## Integration

The workflow engine integrates with existing Cliver components:

- **LLM Integration**: Uses `cliver.llm.TaskExecutor` for LLM steps
- **Configuration**: Leverages existing config management
- **CLI Framework**: Built on Click command framework
- **Logging**: Uses standard Python logging infrastructure
