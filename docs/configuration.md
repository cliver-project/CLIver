---
title: Configuration
description: Configure CLIver for LLM providers, MCP servers, and future secrets management
---

# Configuration Guide

This guide covers how to configure CLIver for different LLM providers, Model Context Protocol (MCP) servers, and future secrets management.

## Configuration Overview

CLIver supports flexible configuration through multiple methods:

1. **Configuration File**: JSON file (default: `~/.cliver/config.json`), users can override it by setting `CLIVER_CONF_DIR` environment variable.
2. **Environment Variables**: For sensitive information like API keys

## Basic Configuration File

The default configuration file is located at `~/.cliver/config.json`. You can create this file manually or let CLIver generate it on first run:

```json
{
  "models": {
        "deepseek-r1": {
            "name_in_provider": "deepseek-r1:14b",
            "provider": "openai",
            "api_key": "dummy",
            "url": "http://127.0.0.1:8080"
        },
        "llama3": {
            "name_in_provider": "llama3.2:latest",
            "provider": "ollama",
            "api_key": "dummy",
            "url": "http://127.0.0.1:11434"
        }
    },
    "mcpServers": {
        "time": {
            "args": [
                "mcp-server-time",
                "--local-timezone=Asia/Shanghai"
            ],
            "command": "uvx",
            "env": {},
            "transport": "stdio"
        }
    }
}
```

## LLM Configuration

Besides the basic configuration items for each LLM model, you can define inference options and capabilities:

```json
{
  "models": {
        "deepseek-r1": {
            "name_in_provider": "deepseek-r1:14b",
            "provider": "openai",
            "api_key": "dummy",
            "url": "http://127.0.0.1:8080",
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "llama3": {
            "name_in_provider": "llama3.2:latest",
            "provider": "ollama",
            "api_key": "dummy",
            "url": "http://127.0.0.1:11434",
            "temperature": 0.3,
            "top_p": 0.7
        }
    }
}
```

## Configuration Validation

To validate your configuration file:

```bash
cliver config validate
```

This command will check your configuration file for syntax errors and missing required fields.

## Managing MCP Servers

You can now manage your MCP servers using the top-level `mcp` command:

```bash
# List all configured MCP servers
cliver mcp list

# Add an MCP server
cliver mcp add --name my-server --transport stdio --command uvx --args my-mcp-server

# Update an MCP server
cliver mcp set --name my-server --command npx

# Remove an MCP server
cliver mcp remove --name my-server
```

## Managing LLM Models

You can now manage your LLM models using the top-level `llm` command:

```bash
# List all configured LLM models
cliver llm list

# Add an LLM model
cliver llm add --name my-model --provider ollama --url http://localhost:11434 --name-in-provider llama3.2:latest

# Update an LLM model
cliver llm set --name my-model --provider vllm

# Remove an LLM model
cliver llm remove --name my-model
```

## Next Steps

Now that you have configured CLIver, check out the [Chat Command Usage](chat.md) to start interacting with LLMs, or read about [Workflow Definition](workflow.md) to learn how to define complex operations.