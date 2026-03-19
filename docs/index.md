# Welcome to CLIver Documentation

CLIver is a command-line interface (CLI) and Python library designed to provide seamless integration with large language models (LLMs) and Model Context Protocol (MCP) servers. The project emphasizes flexibility, extensibility, and secure operations, making it suitable for both interactive use and programmatic integration.

## Getting Started

To start using CLIver, we recommend following these steps:

1. [Overview](overview.md) - Introduction to CLIver and its design goals
2. [Installation Guide](installation.md) - How to install and set up CLIver
3. [Configuration](configuration.md) - Configure CLIver for LLM providers and MCP servers
4. [Chat Command Usage](chat.md) - Learn how to use the `cliver chat` command
5. [Skills](skills.md) - LLM-driven skill activation for specialized tasks
6. [Memory & Identity](memory-identity.md) - Agent memory, identity profiles, and multi-agent isolation
7. [Session Management](session-management.md) - Conversation sessions, history, and compression
8. [Permissions](permissions.md) - Control tool execution permissions and resource access
9. [Workflow Definition](workflow.md) - Define and execute complex workflows
10. [Extensibility Guide](extensibility.md) - Extend CLIver functionality and use as a Python library
11. [Roadmap](roadmap.md) - Future development plans and contribution guidelines

## Quick Start

Install CLIver with pip:

```bash
pip install cliver
```

Start a chat session:

```bash
cliver
```
> This will start an interactive CLI in which you can start the sub commands using slash(`/`) like `/mcp`, etc.

Manage MCP servers:

```bash
cliver mcp list
cliver mcp add --name my-server --transport stdio --command uvx
```

Manage LLM models:

```bash
cliver model list
cliver model add --name my-model --provider ollama --url http://localhost:11434
cliver model add --name deepseek --provider openai --url http://192.168.1.100:8080
```

## Key Features

- **Multi-LLM Support**: Connect to various language models served by various providers (DeepSeek, OpenAI, Qwen3-coder on OpenAI compatible servers, vLLM, and more in the future)
- **MCP Integration**: Seamless integration with Model Context Protocol servers for enhanced functionality
- **Skills**: LLM-driven skill activation from SKILL.md files for specialized tasks
- **Memory & Identity**: Agent memory (append/rewrite) and identity profiles with multi-agent isolation
- **Session Management**: Conversation history with LLM-based compression and session persistence
- **Tool Permissions**: Resource-aware permission system with persistent rules, session grants, and workflow-scoped overrides
- **Configurable Workflows**: Define and execute complex workflows using YAML configuration files
- **Task Scheduling**: Schedule workflows with cron expressions
- **Token Usage**: Track and view token usage statistics per model and session
- **Extensible Architecture**: Easy to extend with custom commands and backends — usable as both a CLI and a Python library
- **Secure Operations**: Secrets management via keyring integration and secure API key handling
