# CLIver

CLIver is another AI agent to make your CLI clever and more.

## Documentation

For complete documentation, visit the documentation pages:

- [Overview](docs/overview.md) - Introduction to CLIver and its design goals
- [Installation Guide](docs/installation.md) - How to install and set up CLIver
- [Configuration](docs/configuration.md) - Configure CLIver for LLM providers and MCP servers
- [Chat Command Usage](docs/chat.md) - Learn how to use the `cliver chat` command
- [Workflow Definition](docs/workflow.md) - Define and execute complex workflows
- [Extensibility Guide](docs/extensibility.md) - Extend CLIver functionality and use as a Python library
- [Roadmap](docs/roadmap.md) - Future development plans and contribution guidelines

## Overview
CLIver is an AI-powered command-line interface tool that enhances your terminal experience with intelligent capabilities.
It integrates with MCP (Model Coordination Protocol) servers and various LLM providers to provide an interactive CLI experience.

## Features
- Interactive and batch mode chat with LLM models
- Integration with MCP servers for extended tool capabilities
- Support for multiple LLM providers (Ollama, OpenAI, vLLM)
- Extensible command system
- Skill sets and templates for prompt enhancement
- Multi-media support (images, audio, video)

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
- `-ss, --skill-set`: Apply skill sets to the chat session
- `-t, --template`: Use a template for the prompt
- `-p, --param`: Specify parameters for skill sets and templates (key=value)

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

#### Chat with Skill Sets
```bash
cliver chat -ss "code_review" "Please review this code" -p "FILE_PATH=/path/to/file.py"
```

> CLIver will search for skill set named `code_review` for system messages and apply the parameters in Jinja2 syntax

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
CLIver uses JSON-based configuration stored in the user's config directory. You can manage configuration using various commands:

- Use `config` command to validate, show, or view the configuration path
- Use `mcp` command to manage Model Context Protocol servers  
- Use `llm` command to manage LLM models

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
Use the top-level `llm` command to manage LLM models:

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

## Extending CLIver
New commands can be added by creating Python files in `~/.config/cliver/commands/` (by default) with Click group definitions.
