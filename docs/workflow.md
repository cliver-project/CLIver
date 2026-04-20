---
title: Workflows
description: Multi-step workflow execution with LangGraph and subagents
---

# Workflows

CLIver's workflow engine executes multi-step AI pipelines using [LangGraph](https://github.com/langchain-ai/langgraph) StateGraph. Each LLM step runs as an isolated subagent with its own model, tools, and permissions.

## Quick Start

```yaml
# .cliver/workflows/research.yaml
name: research
description: "Research and summarize a topic"

agents:
  researcher:
    model: deepseek-r1
    system_message: "You are a thorough researcher."
    tools: [read_file, web_search]

steps:
  - id: research
    type: llm
    agent: researcher
    prompt: "Research: {{ inputs.topic }}"
    output_format: md

  - id: summarize
    type: llm
    agent: researcher
    prompt: "Summarize: {{ research.outputs.result }}"
    depends_on: [research]
    output_format: md
```

Run it:
```bash
cliver workflow run research -i topic="quantum computing"
```

## Concepts

### Subagents

Each LLM step is a **subagent** -- a fresh, isolated AgentCore instance. Subagents don't share conversation context but all receive the workflow overview.

### Agent Profiles

Define reusable agent configurations in the `agents:` section:

```yaml
agents:
  architect:
    model: deepseek-r1
    system_message: "You are a software architect."
    tools: [read_file, grep_search, list_directory]
    skills: [brainstorm, write-plan]
    permissions:
      mode: auto-edit
```

Fields:
- `model` -- LLM model to use
- `system_message` -- System prompt for this agent
- `tools` -- Builtin tools to enable (subset of available tools)
- `skills` -- Skills to pre-activate from the shared pool
- `permissions` -- Permission mode and rules

### Workflow Overview

A shared context document injected into every subagent's system prompt:

```yaml
overview: |
  ## Project: Web Dashboard
  ### Directory Structure
  frontend/ -- React + TypeScript
  backend/ -- FastAPI + SQLAlchemy

# Or reference a file:
overview_file: docs/project-overview.md
```

### Step Types

| Type | Description |
|------|-------------|
| `llm` | Run an LLM inference (subagent) |
| `human` | Pause for human input |
| `function` | Call a Python function |
| `decision` | Conditional branching |
| `workflow` | Nested workflow |

### Output Format

Each step can save its output to a file:

```yaml
steps:
  - id: design
    type: llm
    prompt: "Design the feature"
    output_format: md  # md, json, txt, yaml, code
```

Outputs are saved to the run directory (e.g., `.cliver/workflow-runs/{name}/{execution_id}/design.md`).

### Dependencies and Parallelism

Steps run in parallel unless `depends_on` creates ordering:

```yaml
steps:
  - id: start
    type: llm
    prompt: "Begin"

  - id: branch_a
    type: llm
    prompt: "Do A"
    depends_on: [start]    # runs after start

  - id: branch_b
    type: llm
    prompt: "Do B"
    depends_on: [start]    # runs in parallel with branch_a

  - id: merge
    type: llm
    prompt: "Combine: {{ branch_a.outputs.result }} + {{ branch_b.outputs.result }}"
    depends_on: [branch_a, branch_b]  # waits for both
```

### Variable Propagation

Use Jinja2 templates to pass data between steps:

- `{{ inputs.topic }}` -- workflow input
- `{{ step_id.outputs.result }}` -- output from a previous step

### Pause and Resume

Human steps pause the workflow for input. Resume with:

```bash
cliver workflow resume my-workflow --thread <thread_id> --answer "approved"
```

Workflow state is persisted in SQLite, surviving restarts.

## CLI Commands

```bash
cliver workflow list              # List saved workflows
cliver workflow show <name>       # Show workflow YAML
cliver workflow run <name>        # Execute a workflow
cliver workflow resume <name>     # Resume a paused workflow
cliver workflow delete <name>     # Delete a workflow
```

## Scheduled Workflows

Workflows can be scheduled via the gateway:

```yaml
# Task definition with workflow
name: nightly-review
prompt: "Review today's code changes"
workflow: code-review-pipeline
schedule: "0 22 * * *"
skills: [brainstorm]
```

See [Gateway](docs/gateway.md) for more on task scheduling.

## Example: Feature Development Pipeline

```yaml
name: feature-pipeline
description: "End-to-end feature development"

overview: |
  ## Project: CLIver
  Python CLI agent with modular architecture.
  Use uv for package management, pytest for testing.

agents:
  architect:
    model: deepseek-r1
    system_message: "You are a software architect. Produce clean designs."
    tools: [read_file, grep_search, list_directory]
    skills: [brainstorm, write-plan]
    permissions:
      mode: auto-edit

  developer:
    model: qwen
    system_message: "You are a senior developer. Write clean, tested code."
    tools: [read_file, write_file, run_shell_command, grep_search]
    permissions:
      mode: auto-edit
      rules:
        - tool: run_shell_command
          resource: "git *"
          action: allow

inputs:
  requirements: ""

steps:
  - id: design
    type: llm
    agent: architect
    prompt: "Design a feature based on: {{ inputs.requirements }}"
    output_format: md

  - id: review
    type: human
    prompt: "Review the design and approve or provide feedback"
    depends_on: [design]

  - id: implement
    type: llm
    agent: developer
    prompt: |
      Implement based on this design:
      {{ design.outputs.result }}
      Feedback: {{ review.outputs.result }}
    depends_on: [review]
    output_format: md

  - id: test
    type: llm
    agent: developer
    prompt: "Write tests for the implementation"
    depends_on: [implement]
    output_format: md
```
