---
title: Workflows
description: Define and execute complex workflows using YAML configuration files
---

# Workflow Definition

CLIver's workflow system allows you to define and execute complex multi-step operations using YAML configuration files. Workflows enable automation of repeated tasks, complex interactions, and integration between different LLMs and tools.

## Workflow Structure

A workflow is defined in a YAML file with the following top-level keys:

- `name`: A descriptive name for the workflow
- `description`: A brief description of what the workflow does
- `inputs`: Input parameters for the workflow
- `steps`: An ordered list of steps to execute
- `outputs`: How to handle the final output

## Basic Workflow Example

Here's a simple workflow that summarizes a document:

```yaml
name: Document Summarizer
description: Summarizes a document using an LLM
inputs:
  document_path:
    type: string
    description: Path to the document to summarize
    required: true
  model:
    type: string
    description: Model to use for summarization
    default: "gpt-4-turbo"

steps:
  - name: read_document
    action: file.read
    parameters:
      path: "{{ inputs.document_path }}"

  - name: summarize
    action: llm.chat
    parameters:
      model: "{{ inputs.model }}"
      messages:
        - role: system
          content: "Provide a concise summary of the following document."
        - role: user
          content: "{{ steps.read_document.output }}"
      temperature: 0.3
      max_tokens: 500

outputs:
  summary: "{{ steps.summarize.output }}"
```

## Workflow Steps

Each step in a workflow has the following properties:

- `name`: A unique identifier for the step
- `action`: The action to perform (e.g., `llm.chat`, `file.read`, `mcp.call`)
- `parameters`: Parameters to pass to the action
- `condition`: (Optional) A condition to determine if the step should execute

### Available Actions

#### llm.chat
Interact with a language model:

```yaml
- name: generate_response
  action: llm.chat
  parameters:
    model: gpt-4-turbo
    messages:
      - role: system
        content: "You are a helpful assistant."
      - role: user
        content: "What are the key points in: {{ document_content }}"
    temperature: 0.7
    max_tokens: 1000
```

#### file.read
Read a file from the filesystem:

```yaml
- name: read_input
  action: file.read
  parameters:
    path: "{{ inputs.input_file }}"
    encoding: "utf-8"
```

#### file.write
Write content to a file:

```yaml
- name: save_output
  action: file.write
  parameters:
    path: "{{ inputs.output_file }}"
    content: "{{ steps.process.output }}"
    encoding: "utf-8"
```

#### mcp.call
Call an MCP server:

```yaml
- name: get_context
  action: mcp.call
  parameters:
    server: "local-mcp-server"
    operation: "getResource"
    parameters:
      type: "secret"
      name: "api-key"
```

## Variables and Templates

CLIver uses a template system to reference values from different parts of the workflow:

- `{{ inputs.variable_name }}`: Reference an input parameter
- `{{ steps.step_name.output }}`: Reference the output of a previous step
- `{{ constants.value }}`: Reference a constant value

### Example with Variables

```yaml
name: Code Review Workflow
description: Reviews code and provides suggestions
inputs:
  code_file:
    type: string
    description: Path to the code file to review
    required: true
  model:
    type: string
    description: Model to use for code review
    default: "gpt-4-turbo"

steps:
  - name: read_code
    action: file.read
    parameters:
      path: "{{ inputs.code_file }}"

  - name: review_code
    action: llm.chat
    parameters:
      model: "{{ inputs.model }}"
      messages:
        - role: system
          content: |
            You are an expert code reviewer. Identify potential bugs, 
            performance issues, security vulnerabilities, and suggest improvements.
        - role: user
          content: |
            Review this code and provide feedback:
            ```{{ get_file_extension(inputs.code_file) }}
            {{ steps.read_code.output }}
            ```
      temperature: 0.3
      max_tokens: 1500

  - name: format_review
    action: llm.chat
    parameters:
      model: "{{ inputs.model }}"
      messages:
        - role: system
          content: "Format the code review into a structured markdown report."
        - role: user
          content: "{{ steps.review_code.output }}"
      temperature: 0.2

outputs:
  review_report: "{{ steps.format_review.output }}"
  original_code: "{{ steps.read_code.output }}"
```

## Conditional Steps

Use conditions to make workflows more flexible:

```yaml
steps:
  - name: analyze_sentiment
    action: llm.chat
    parameters:
      model: gpt-4-turbo
      messages:
        - role: user
          content: "Analyze the sentiment of: {{ inputs.text }}"
  
  - name: positive_feedback
    action: llm.chat
    condition: "sentiment == 'positive'"
    parameters:
      model: gpt-4-turbo
      messages:
        - role: system
          content: "Provide encouragement based on positive sentiment."
        - role: user
          content: "{{ inputs.text }}"
  
  - name: improvement_suggestions
    action: llm.chat
    condition: "sentiment == 'negative'"
    parameters:
      model: gpt-4-turbo
      messages:
        - role: system
          content: "Provide constructive suggestions for improvement."
        - role: user
          content: "{{ inputs.text }}"
```

## Loops in Workflows

For processing multiple items, you can use loops:

```yaml
name: Multi-Document Processing
description: Process multiple documents in sequence
inputs:
  document_paths:
    type: array
    description: List of document paths to process
    required: true

steps:
  - name: process_documents
    action: loop
    parameters:
      items: "{{ inputs.document_paths }}"
      step:
        name: single_document
        action: llm.chat
        parameters:
          model: gpt-4-turbo
          messages:
            - role: system
              content: "Summarize the following document."
            - role: user
              content: "{{ item }}"
          temperature: 0.3
          max_tokens: 300

outputs:
  summaries: "{{ steps.process_documents.output }}"
```

## Local Directory Implementation

Workflows can be organized in local directories for better management:

```
workflows/
├── general/
│   ├── document-processor.yaml
│   └── code-reviewer.yaml
├── business/
│   ├── report-generator.yaml
│   └── email-responder.yaml
└── development/
    ├── pull-request-reviewer.yaml
    └── bug-analyzer.yaml
```

To execute a workflow from a specific directory:

```bash
cliver workflow run --path workflows/general/document-processor.yaml --input document_path=/path/to/doc.txt
```

## Running Workflows

### From Command Line

```bash
# Run a workflow with input parameters
cliver workflow run --path /path/to/workflow.yaml --input document_path=/path/to/doc.txt --input model=gpt-4-turbo

# Run with inputs from a JSON file
cliver workflow run --path /path/to/workflow.yaml --inputs-file /path/to/inputs.json

# Dry run to validate the workflow without executing
cliver workflow run --path /path/to/workflow.yaml --dry-run
```

### From Python Library

```python
from cliver import WorkflowRunner

runner = WorkflowRunner()
result = runner.execute_workflow(
    workflow_path="/path/to/workflow.yaml",
    inputs={
        "document_path": "/path/to/doc.txt",
        "model": "gpt-4-turbo"
    }
)
print(result)
```

## Built-in Workflows

CLIver comes with several built-in workflows for common tasks:

- `document-summarizer`: Summarizes documents
- `code-reviewer`: Reviews code for issues and improvements
- `email-responder`: Generates email responses
- `report-builder`: Creates structured reports
- `content-analyzer`: Analyzes content for sentiment, keywords, etc.

To list available built-in workflows:

```bash
cliver workflow list --built-in
```

## Error Handling

Workflows can define error handling behavior:

```yaml
name: Robust Document Processor
description: Processes documents with error handling
inputs:
  document_path: ...

steps:
  - name: read_document
    action: file.read
    parameters:
      path: "{{ inputs.document_path }}"
    on_error:
      fallback: "Could not read document, using default content"
      retry: 3
      retry_delay: 5

  - name: process_document
    action: llm.chat
    parameters:
      model: gpt-4-turbo
      messages:
        - role: user
          content: "Process this: {{ steps.read_document.output }}"
    on_error:
      fallback_action: error_handler

  - name: error_handler
    action: llm.chat
    condition: "error_occurred"
    parameters:
      model: gpt-4-turbo
      messages:
        - role: system
          content: "Generate a helpful error message."
        - role: user
          content: "An error occurred while processing. Provide a user-friendly error message."
```

## Next Steps

Now that you understand how to define workflows, explore how to [extend CLIver](extensibility.md) with custom actions and components, or check out the [roadmap](roadmap.md) for upcoming features.