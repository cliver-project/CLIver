# CLIver
Yet another AI agent to make your CLI clever

## Overview
CLIver is an AI-powered command-line interface tool that enhances your terminal experience with intelligent capabilities.
It integrates with MCP (Model Coordination Protocol) servers and various LLM providers to provide an interactive CLI experience.

## Features
- Interactive chat with LLM models
- Integration with MCP servers for extended tool capabilities
- Support for multiple LLM providers (Ollama, OpenAI, etc.)
- Extensible command system
- Skill sets and templates for prompt enhancement

## Installation
```bash
uv venv
uv sync --all-extras --dev --locked
```

## Usage
```bash
uv run cliver
```

### Chat Command
The main way to interact with CLIver is through the chat command:

```bash
cliver chat "Your question here"
```

#### Options
- `-m, --model`: Specify which LLM model to use
- `-s, --stream`: Stream the response
- `-ss, --skill-set`: Apply skill sets to the chat session
- `-t, --template`: Use a template for the prompt
- `-p, --param`: Specify parameters for skill sets and templates (key=value)

### Examples

#### Basic Chat
```bash
cliver chat "What is the capital of France?"
```

#### Chat with Specific Model
```bash
cliver chat -m "gpt-4" "Explain quantum computing"
```

#### Chat with Streaming
```bash
cliver chat -s "Write a poem about programming"
```

#### Chat with Skill Sets
```bash
cliver chat -ss "code_review" "Please review this code" -p "FILE_PATH=/path/to/file.py"
```

#### Chat with Templates
```bash
cliver chat -t "code_review_template" "Please review this code" -p "code=def hello(): print('Hello')"
```

#### Chat with Both Skill Sets and Templates
```bash
cliver chat -ss "code_review" -t "code_review_template" "Please review this code" -p "FILE_PATH=/path/to/file.py" -p "code=def hello(): print('Hello')"
```

## Skill Sets
Skill sets are predefined collections of capabilities, tools, and context that can be applied to a chat session. They are defined in YAML files.

### Example Skill Set
```yaml
description: File system operations
system_message: |
  You are an expert file system assistant. You have access to tools for
  file operations. When asked to perform operations, be precise and
  confirm destructive actions.
tools:
  - name: read_file
    mcp_server: file_system
    description: Read the contents of a file
    parameters:
      path: ${file_path}
  - name: write_file
    mcp_server: file_system
    description: Write content to a file
    parameters:
      path: ${file_path}
      content: ${content}
parameters:
  file_path: /default/path.txt
  content: Default content
```

## Templates
Templates are predefined prompt structures with placeholders that can be filled with user-provided values. They are defined in text files.

### Example Template
```
Please review the following code for quality, security, and best practices:

{code}

Focus on:
1. Code readability and maintainability
2. Security vulnerabilities
3. Performance issues
4. Best practices and coding standards

Provide specific suggestions for improvement.
```

## Configuration
CLIver uses JSON-based configuration stored in the user's config directory. You can manage configuration using the `config` command.

## Extending CLIver
New commands can be added by creating Python files in `src/cliver/commands/` with Click group definitions.

## Development
### Setup
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