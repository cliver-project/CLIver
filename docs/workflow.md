---
title: Workflows
description: Define and execute complex workflows using YAML configuration files
---

# Workflow Definition
CLIver has a simple workflow engine to allow you to define and execute multi-step operations using YAML files.
Each step can be one of the following `4` types:

- `llm`:        LLM inference request with defined inputs
- `function`:   Python function calls
- `human`:      Human interaction to confirm before going to next step
- `workflow`:   Call other workflow for a complex sub task

## Workflow Structure

A workflow is defined in a YAML file with the following top-level keys:

- `name`: A descriptive name for the workflow (must be globally unique)
- `description`: A brief description of what the workflow does
- `version`: Version of the workflow (optional)
- `author`: Author of the workflow (optional)
- `inputs`: List of input variable names for the workflow (optional)
- `steps`: An ordered list of steps to execute

## Basic Workflow Example

Here's a simple workflow that analyzes a topic:

```yaml
--8<-- "examples/basic_workflow.yaml"
```


## Workflow Steps

Each step in a workflow has the following properties:

- `id`: A unique identifier for the step (required)
- `name`: A descriptive name for the step (required)
- `type`: The type of step (required, one of: `llm`, `function`, `human`, `workflow`)
- `description`: A description of what the step does (optional)
- `inputs`: Input variables for the step (optional)
- `outputs`: Output variable names from the step (optional)
- `retry`: Retry policy configuration (optional)
- `timeout`: Timeout in seconds (optional)
- `on_error`: Action to take on error (optional, `fail` or `continue`)
- `condition`: Condition expression for step execution (optional)
- `skipped`: Whether the step is skipped (optional, default: false)

### Available Actions

#### LLM Steps
Interact with a language model:

```yaml
--8<-- "examples/llm_step.yaml"
```

LLM steps support the following properties:
- `prompt`: The prompt to send to the LLM
- `model`: The LLM model to use (optional, defaults to the model configured in your CLIver setup, e.g., 'deepseek-r1' or 'qwen')
- `stream`: Whether to stream the response (optional, default: false)
- `images`: Image files to send with the message (optional)
- `audio_files`: Audio files to send with the message (optional)
- `video_files`: Video files to send with the message (optional)
- `files`: General files to upload for tools (optional)
- `skill_sets`: Skill sets to apply (optional)
- `template`: Template to use for the prompt (optional)
- `params`: Parameters for skill sets and templates (optional)

#### Function Steps
Execute Python functions:

```yaml
--8<-- "examples/function_step.yaml"
```


Function steps support the following properties:
- `function`: Module path to the function to execute (required)

#### Human Steps
Wait for human interaction:

--8<-- "examples/human_step.yaml"

Human steps support the following properties:
- `prompt`: Prompt to show to the user (required)
- `auto_confirm`: Automatically confirm without user input (optional, default: false)

#### Workflow Steps
Call other workflows:

--8<-- "examples/workflow_step.yaml"

Workflow steps support the following properties:
- `workflow`: Workflow name or path to the workflow file to execute (required)
- `workflow_inputs`: Inputs for the sub-workflow (optional)

## Variables and Templates

CLIver uses a Jinja2-based template system to reference values from different parts of the workflow:

- `{{ inputs.variable_name }}`: Reference an input parameter
- `{{ step_id.output_name }}`: Reference the output of a previous step by step ID and output name
- `{{ step_id.outputs.output_name }}`: Alternative way to reference step outputs
- Environment variables are also available in templates

### Example with Variables

```yaml
--8<-- "examples/code_review_workflow.yaml"
```

## Conditional Steps

Use conditions to make workflows more flexible. Conditions are evaluated using Jinja2 templating:

```yaml
--8<-- "examples/conditional_step.yaml"
```

Note: The current implementation has basic support for conditions, but full conditional branching (if/else) is not yet fully implemented.

## Loops in Workflows

Note: Loop support is not yet fully implemented in the current workflow engine. To process multiple items, you can create separate steps for each item or use LLM steps with prompts that handle multiple items at once.

## Local Directory Implementation

Workflows can be organized in local directories for better management. CLIver looks for workflows in the following directories:

1. `.cliver/workflows` in the current directory
2. `~/.config/cliver/workflows` (user configuration directory)

```
.cliver/
└── workflows/
    ├── example_workflow.yaml
    ├── code_review_workflow.yaml
    └── research_workflow.yaml
```

Workflows are identified by their `name` field, not by their file name. You can organize workflow files in subdirectories as needed.

## Running Workflows

### From Command Line

```bash
# Run a workflow with input parameters
cliver workflow run workflow_name -i document_path=/path/to/doc.txt -i model=deepseek-r1
# Can also use -i model=qwen as an alternative

# Dry run to validate the workflow without executing
cliver workflow run workflow_name --dry-run

# Resume a paused workflow execution
cliver workflow run workflow_name -e execution_id
```

### From Python Library

```python
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager
from cliver.workflow.workflow_executor import WorkflowExecutor
import asyncio

# Create workflow manager and executor (requires a TaskExecutor instance)
# workflow_manager = LocalDirectoryWorkflowManager()
# workflow_executor = WorkflowExecutor(task_executor, workflow_manager)

# Execute workflow (in an async context)
# result = await workflow_executor.execute_workflow(
#     workflow_name="workflow_name",
#     inputs={
#         "document_path": "/path/to/doc.txt",
#         "model": "deepseek-r1"  # Can also use "qwen" as an alternative
#     }
# )
# print(result)
```

## Available Workflows

To list available workflows in your configured workflow directories:

```bash
cliver workflow list
```

Workflows are loaded from the following directories:
1. `.cliver/workflows` in the current directory
2. `~/.config/cliver/workflows` (user configuration directory)

You can place your workflow YAML files in either of these directories to make them available.

## Error Handling

Workflows can define error handling behavior using retry policies and on_error actions:

--8<-- "examples/error_handling.yaml"

Error handling properties:
- `retry`: Retry policy configuration with `max_attempts`, `backoff_factor`, and `max_backoff`
- `on_error`: Action to take on error (`fail` or `continue`)

## Next Steps

Now that you understand how to define workflows, explore how to [extend CLIver](extensibility.md) with custom actions and components, or check out the [roadmap](roadmap.md) for upcoming features.