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
- `inputs`: List of input parameters for the workflow with metadata (optional)
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

## Variable Access in Workflows

Workflows support a powerful variable access system that allows steps to reference data from previous steps, workflow inputs, and other sources. Variables are accessed using Jinja2 templating syntax with specific formats for different types of data.

### Variable Access Formats

CLIver provides a clean and organized way to access variables in workflows:

1. **Workflow Inputs**: Access workflow-level input parameters
   ```yaml
   inputs:
     user_name: "{{ inputs.user_name }}"
   ```

2. **Step Outputs**: Access outputs from previous steps
   ```yaml
   inputs:
     analysis_result: "{{ previous_step.outputs.result }}"
   ```

3. **Step Inputs**: Access inputs from previous steps
   ```yaml
   inputs:
     original_greeting: "{{ previous_step.inputs.greeting }}"
   ```

### Example: Using Variable Access

Here's a complete example showing how to use the variable access formats:

```yaml
--8<-- "examples/variable_access_example.yaml"
```

### Best Practices

1. Use the explicit variable access formats for clarity and maintainability
2. Always specify which step and which type of data you're accessing
3. Test your workflows to ensure variable references resolve correctly
4. Use descriptive step IDs to make variable references more readable

Each input parameter supports the following properties:

- `name`: The name of the input parameter (required)
- `description`: A description of the parameter's purpose (optional)
- `type`: The expected type of the parameter (optional)
- `default`: A default value to use if none is provided (optional)

When executing a workflow, if an input parameter has a default value defined and no value is provided, the default will be used.
If no default is specified and no value is provided, the parameter will be set to `None`.

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

```yaml
--8<-- "examples/human_step.yaml"
```

Human steps support the following properties:

- `prompt`: Prompt to show to the user (required)
- `auto_confirm`: Automatically confirm without user input (optional, default: false)

#### Workflow Steps
Call other workflows:

```yaml
--8<-- "examples/workflow_step.yaml"
```

Workflow steps support the following properties:

- `workflow`: Workflow name or path to the workflow file to execute (required)
- `workflow_inputs`: Inputs for the sub-workflow (optional)

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
cliver workflow run workflow_name -i document_path=/path/to/doc.txt
# Can also use -i model=qwen as an alternative

# Dry run to validate the workflow without executing
cliver workflow run workflow_name --dry-run

# Resume a paused workflow execution
cliver workflow run workflow_name -e execution_id
```

### From Python Library

```python
--8<-- "examples/run_workflow.py"
```

## Available Workflows

To list available workflows in your configured workflow directories:

```bash
cliver workflow list
```

> NOTE: In current implementation, it uses local files based for the workflow definition and execution cache.

Workflows are loaded from the following directories:

1. `.cliver/workflows` in the current directory
2. `~/.config/cliver/workflows` (user configuration directory)

You can place your workflow YAML files in either of these directories to make them available.

## Error Handling

Workflows can define error handling behavior using retry policies and on_error actions:

```yaml
--8<-- "examples/error_handling.yaml"
```


Error handling properties:

- `retry`: Retry policy configuration with `max_attempts`, `backoff_factor`, and `max_backoff`
- `on_error`: Action to take on error (`fail` or `continue`)

## Next Steps

Now that you understand how to define workflows, explore how to [extend CLIver](extensibility.md) with custom actions and components, or check out the [roadmap](roadmap.md) for upcoming features.