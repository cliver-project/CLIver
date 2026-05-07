---
title: Home
description: CLIver — a general-purpose AI agent for your terminal
---

# CLIver Documentation

CLIver is a general-purpose AI agent for your terminal. It is model-agnostic, extensible, and safe by design.

## Getting Started

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**

    Install CLIver via pip, Docker, or from source.

- :material-cog: **[Configuration](configuration.md)**

    Set up LLM providers, MCP servers, and permissions.

- :material-chat: **[Chat Usage](chat.md)**

    Learn interactive and batch chat modes.

- :material-lightning-bolt: **[Skills](skills.md)**

    Teach CLIver new domains with skill files.

</div>

## Core Guides

| Guide | What You'll Learn |
|-------|-------------------|
| [Memory & Identity](memory-identity.md) | Persistent knowledge and agent profiles |
| [Permissions](permissions.md) | Control what tools CLIver can execute |
| [Workflows](workflow.md) | Multi-step task orchestration with LangGraph |
| [Session Management](session-management.md) | Conversation history and compression |
| [Gateway](gateway.md) | Deploy with Telegram, Discord, Slack, Feishu |

## Extend & Integrate

| Guide | What You'll Learn |
|-------|-------------------|
| [Extensibility & API](extensibility.md) | Use AgentCore as a Python library |
| [Roadmap](roadmap.md) | Planned features and how to contribute |

## Quick Example

```bash
# Install and start
pip install cliver
cliver

# Add a model
cliver model add --name deepseek --provider openai --url https://api.deepseek.com

# Add an MCP server
cliver mcp add --name filesystem --transport stdio --command uvx -- mcp-server-filesystem

# Chat with tools
cliver "List all Python files in this directory and summarize the largest one"
```
