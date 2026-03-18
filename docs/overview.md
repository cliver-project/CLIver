---
title: Overview
description: Introduction to CLIver - your powerful command-line and library tool for LLM interactions
---

# CLIver Overview

CLIver is an AI-powered command-line interface tool that enhances your terminal experience with intelligent capabilities.
It integrates with MCP (Model Context Protocol) servers and various LLM providers to provide an interactive CLI experience.

## Design Goals

CLIver is built as a **dual-layer system**:

- **API Layer** (`TaskExecutor`): The core engine — embeddable in any Python application. It handles LLM inference, tool calling, Re-Act loops, permissions, and workflow execution with no dependency on CLI concerns (no terminal, no stdin, no prompt_toolkit). This is the layer you use when integrating CLIver as a library.
- **CLI Layer** (`Cliver` class + Click commands): A thin interactive shell on top of `TaskExecutor` for terminal users. Provides Rich-formatted output, prompt_toolkit input, and slash commands.

Any feature built at the API layer (permissions, workflows, skills, memory) works identically whether invoked from the CLI or from your own Python code.

## Key Features

### Core Capabilities
- **Multi-LLM Support**: Connect to various language models served by various providers(DeepSeek, OpenAI, Qwen3-coder on OpenAI compatible servers, vLLM, and more in the future)
- **MCP Integration**: Seamlessly integrate with Model Context Protocol servers for enhanced functionality
- **Tool Permissions**: Resource-aware [permission system](permissions.md) controlling which tools can execute and what resources they can access
- **Configurable Workflows**: Define and execute complex workflows using YAML configuration files
- **Extensible Architecture**: Easy to extend with custom commands and backends

### Usage Modes
CLIver operates in two primary modes:

1. **Interactive and Batch CLI Mode**: Direct command-line interaction for immediate responses and operations
2. **Library Mode**: Python library integration for embedding LLM capabilities in your applications

## Architecture

CLIver follows a modular architecture that allows for easy extension:

<figure markdown>

```mermaid
graph TD
    A[CLIver Class] --> B[ConfigManager]
    A --> C[TaskExecutor]
    A --> D[CLI Interface]

    B --> E[LLM Model Config]
    B --> F[MCP Server Config]

    C --> G[LLM Inference Engines]
    C --> H[MCP & Builtin Tools Integration]
    C --> I[Workflow Engine]
    C --> N[Permission Manager]

    G --> J[Ollama Provider]
    G --> K[OpenAI Provider]
    G --> L[Other LLM Providers]

    H --> M[MCPServersCaller]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style G fill:#e8f5e8
    style H fill:#e8f5e8
    style I fill:#e8f5e8
```

</figure>

## Getting Started

To start using CLIver, visit our [Installation Guide](installation.md) to set up the tool on your system, followed by the [Configuration Guide](configuration.md) to connect to your preferred LLM provider.