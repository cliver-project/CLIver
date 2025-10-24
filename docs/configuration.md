---
title: Configuration
description: Configure CLIver for LLM providers, MCP servers, and future secrets management
---

# Configuration Guide

This guide covers how to configure CLIver for different LLM providers, Model Context Protocol (MCP) servers, and secrets management.

## Configuration Overview

CLIver supports flexible configuration through multiple methods:

1. **Configuration File**: JSON file (default: `~/.cliver/config.json`), users can override it by setting `CLIVER_CONF_DIR` environment variable.
2. **Environment Variables**: For sensitive information like API keys

## Basic Configuration File

The default configuration file is located at `~/.cliver/config.json`. You can create this file manually or let CLIver generate it on first run:

```json
--8<-- "examples/cliver_config.json"
```

### Sample Configuration Explanation

The sample configuration file above demonstrates the key components of CLIver's configuration:

- **models**: Defines the LLM models available to CLIver
  - `deepseek-r1`: An example using the OpenAI provider with:
    - `name_in_provider`: The model name as known to the provider
    - `provider`: The LLM provider type (openai, ollama, etc.)
    - `api_key`: API key retrieved from environment variables using Jinja2 templating
    - `url`: Endpoint for the LLM service
  - `llama3`: An example using the Ollama provider with:
    - `name_in_provider`: The model name as known to Ollama
    - `provider`: Specifies the Ollama provider
    - `api_key`: API key retrieved from system keyring using Jinja2 templating
    - `url`: Endpoint for the Ollama service

- **mcpServers**: Configures Model Context Protocol (MCP) servers that extend CLIver's capabilities
  - `time`: An example MCP server for time-related queries with:
    - `args`: Command-line arguments for the MCP server
    - `command`: The command to execute the MCP server
    - `env`: Environment variables for the MCP server (empty in this example)
    - `transport`: Communication method (stdio for standard input/output)

#### Template Support and Secrets Management

CLIver supports Jinja2 templating in configuration files for dynamic value resolution:

- **Environment Variables**: Use `{{ env.VARIABLE_NAME }}` syntax to reference environment variables
- **Keyring Storage**: Use `{{ keyring('KEY_NAME') }}` syntax to retrieve secrets from the system keyring
- **Default Values**: Provide fallback values using the pipe operator, e.g., `{{ env.OPENAI_API_KEY | 'dummy' }}`

This approach keeps sensitive information secure while allowing flexible configuration management.


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