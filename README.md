# CLIver

CLIver is another AI agent to make your CLI clever and more.

## Documentation

For complete documentation, visit the documentation pages:

- [Overview](docs/overview.md) - Introduction to CLIver and its design goals
- [Installation Guide](docs/installation.md) - How to install and set up CLIver
- [Configuration](docs/configuration.md) - Configure CLIver for LLM providers and MCP servers
- [Chat Command Usage](docs/chat.md) - Learn how to use the `cliver chat` command
- [Skills](docs/skills.md) - LLM-driven skill activation for specialized tasks
- [Memory & Identity](docs/memory-identity.md) - Agent memory, identity profiles, and multi-agent isolation
- [Session Management](docs/session-management.md) - Conversation sessions, history, and compression
- [Permissions](docs/permissions.md) - Control tool execution permissions and resource access
- [Workflow Definition](docs/workflow.md) - Define and execute complex workflows
- [Extensibility Guide](docs/extensibility.md) - Extend CLIver functionality and use as a Python library
- [Roadmap](docs/roadmap.md) - Future development plans and contribution guidelines

## Overview
CLIver is an AI-powered command-line interface tool that enhances your terminal experience with intelligent capabilities.
It integrates with MCP (Model Context Protocol) servers and various LLM providers to provide an interactive CLI experience.

## Features
- Interactive and batch mode chat with LLM models
- Integration with MCP servers for extended tool capabilities
- Support for multiple LLM providers (Ollama, OpenAI, vLLM)
- Extensible command system
- LLM-driven skill activation for specialized tasks
- Multi-media support (images, audio, video)
- Agent memory and identity profiles
- Conversation session management with history and compression
- Layered tool permission system (default, auto-edit, yolo modes)
- Token usage statistics and cost tracking
- Workflow engine with pause/resume support
- Task scheduling with cron expressions
- Keyring-based secret management for API keys

## Development

* Set up development environment
```bash
uv venv
uv sync --all-extras --dev --locked
```


### Running Tests
```bash
uv run pytest
```

### Code Quality
```bash
# Format code
uv run ruff format

# Check formatting
uv run ruff format --check

# Linting
uv run ruff check
```

## Start the CLIver application
```bash
uv run cliver
```

### Chat Command
The main way to interact with CLIver is through the chat command:

> After installation, you will have the cliver executable ready.

```bash
cliver chat "Your question here"
```

#### Options
- `-m, --model`: Specify which LLM model to use
- `-s, --stream`: Stream the response
- `-img, --image`: Image files to send with the message
- `-aud, --audio`: Audio files to send with the message
- `-vid, --video`: Video files to send with the message
- `-f, --file`: Specify files to upload for tools like code interpreter
- `-t, --template`: Use a template for the prompt
- `-p, --param`: Specify parameters for templates (key=value)
- `--system-message`: Append a system message
- `--included-tools`: Filter tools by pattern

### Examples

#### Basic Chat with default model
```bash
cliver chat "What is the capital of China?"
```

#### Chat with Specific Model
```bash
cliver chat -m "deepseek" "Please tell me what time is it now in Beijing, Tokyo and London."
```

#### Chat with Streaming
```bash
cliver chat -s -m "deepseek" "Write a poem about programming"
```

#### Chat with Templates
```bash
cliver chat -t "code_review_template" "Please review this code" -p "code=def hello(): print('Hello')"
```
> CLIver will search for a template file `code_review_template.md[.txt]` for user message and apply the parameters in Jinja2 syntax

### Supported LLM Providers
- OpenAI-compatible models (Qwen-coder, Qwen-VL, DeepSeek, etc)
- Ollama models (llama, etc.)
- vLLM models

## Configuration
CLIver uses YAML-based configuration stored in the user's config directory (`~/.config/cliver/config.yaml`). You can manage configuration using various commands:

- Use `config` command to validate, show, or view the configuration path
- Use `mcp` command to manage Model Context Protocol servers
- Use `model` command to manage LLM models

### Managing MCP Servers
Use the top-level `mcp` command to manage MCP servers:

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

### Managing LLM Models
Use the top-level `model` command to manage LLM models:

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

## All Commands

| Command | Description |
|---------|-------------|
| `chat` | Interactive and batch mode chat with LLMs (default command) |
| `model` | Manage LLM model configurations |
| `mcp` | Manage MCP server configurations |
| `config` | Show, validate, and update general settings |
| `workflow` | Define and execute multi-step workflows |
| `task` | Manage agent tasks (workflow + scheduling) |
| `session` | Manage conversation sessions (list, load, new, delete, compress) |
| `session-option` | Manage persistent inference options for interactive sessions |
| `permissions` | Manage persistent permission rules |
| `identity` | Manage agent identity profile |
| `cost` | View token usage statistics |
| `capabilities` | Display model capabilities matrix |
| `help` | Show help for commands |

## Extending CLIver
New commands can be added by creating Python files in `~/.config/cliver/commands/` (by default) with Click group definitions.
