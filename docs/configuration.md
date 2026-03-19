---
title: Configuration
description: Configure CLIver for LLM providers, MCP servers, and secrets management
---

# Configuration Guide

This guide covers how to configure CLIver for different LLM providers, Model Context Protocol (MCP) servers, and secrets management.

## Configuration Overview

CLIver supports flexible configuration through multiple methods:

1. **Configuration File**: YAML file (default: `~/.config/cliver/config.yaml`), users can override it by setting `CLIVER_CONF_DIR` environment variable.
2. **Environment Variables**: For sensitive information like API keys
3. **Keyring**: System keyring integration for secure secret storage

## Basic Configuration File

The default configuration file is located at `~/.config/cliver/config.yaml`. You can create this file manually or let CLIver generate it on first run:

```yaml
--8<-- "examples/cliver_config.yaml"
```

### Sample Configuration Explanation

The sample configuration file above demonstrates the key components of CLIver's configuration:

- **agent_name**: The display name of the AI agent (default: "CLIver")
- **default_model**: The default LLM model to use when none is specified

- **models**: Defines the LLM models available to CLIver
    - `deepseek-r1`: An example using the OpenAI provider with:
        - `name_in_provider`: The model name as known to the provider
        - `provider`: The LLM provider type (openai, ollama, vllm)
        - `api_key`: API key — supports Jinja2 templating or `keyring:<service>:<key>` syntax
        - `url`: Endpoint for the LLM service
    - `llama3`: An example using the Ollama provider with:
        - `name_in_provider`: The model name as known to Ollama
        - `provider`: Specifies the Ollama provider
        - `api_key`: API key retrieved from system keyring using `keyring:<service>:<key>` syntax
        - `url`: Endpoint for the Ollama service

- **mcpServers**: Configures Model Context Protocol (MCP) servers that extend CLIver's capabilities
    - `time`: An example MCP server for time-related queries with:
        - `args`: Command-line arguments for the MCP server
        - `command`: The command to execute the MCP server
        - `env`: Environment variables for the MCP server
        - `transport`: Communication method (stdio, sse, streamable_http, websocket)

#### Template Support and Secrets Management

CLIver supports Jinja2 templating in configuration files for dynamic value resolution:

- **Environment Variables**: Use `{{ env.VARIABLE_NAME }}` syntax to reference environment variables
- **Keyring Storage**: Use `keyring:<service>:<key>` syntax to retrieve secrets from the system keyring
- **Default Values**: Provide fallback values using the pipe operator, e.g., `{{ env.OPENAI_API_KEY | 'dummy' }}`

This approach keeps sensitive information secure while allowing flexible configuration management.


## Configuration Validation

To validate your configuration file:

```bash
cliver config validate
```

This command will check your configuration file for syntax errors and missing required fields.

## Managing MCP Servers

You can manage your MCP servers using the top-level `mcp` command:

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

You can manage your LLM models using the top-level `model` command:

```bash
# List all configured LLM models
cliver model list

# Add an LLM model
cliver model add --name my-model --provider ollama --url http://localhost:11434 --name-in-provider llama3.2:latest

# Update an LLM model
cliver model set --name my-model --provider vllm

# Remove an LLM model
cliver model remove --name my-model

# Show or set the default model
cliver model default
cliver model default my-model
```

## Configuration Directory Structure

```
~/.config/cliver/
├── config.yaml                  # Main configuration (YAML)
├── cliver-settings.yaml         # Global permission rules
├── audit_logs/                  # Token usage logs
├── memory.md                    # Global memory
├── agents/{agent_name}/
│   ├── identity.md              # Agent identity profile
│   ├── memory.md                # Agent memory
│   ├── sessions/                # Conversation sessions (JSONL)
│   └── tasks/                   # Task definitions
├── skills/                      # Global skills (SKILL.md)
├── workflows/                   # Global workflows (YAML)
└── commands/                    # External commands (Python)

.cliver/                         # Project-local (current directory)
├── cliver-settings.yaml         # Local permission rules
├── skills/                      # Project skills
└── workflows/                   # Project workflows
```

## Next Steps

Now that you have configured CLIver, check out the [Chat Command Usage](chat.md) to start interacting with LLMs, or read about [Skills](skills.md) to learn about LLM-driven skill activation.
