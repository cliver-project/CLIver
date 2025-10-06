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
- Multi-media support (images, audio, video)

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
- `-md, --media`: Specify media files to send with the message
- `-f, --file`: Specify files to upload for tools like code interpreter
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

#### Chat with Media Files
```bash
# Analyze an image
cliver chat --media path/to/image.jpg "What's in this image?"

# Stream response with media
cliver chat --stream --media path/to/image.jpg "Describe this image in detail"

# Send multiple media files
cliver chat --media path/to/image1.jpg --media path/to/image2.jpg "Compare these images"
```

## Multi-Media Support
CLIver now supports sending images, audio, and video files along with text to LLMs. This feature works with compatible models like OpenAI's GPT-4 Vision and Ollama's LLaVA.

### Supported Media Types
- Images (JPEG, PNG, GIF, etc.)
- Audio files (WAV, MP3, etc.) - sent as text descriptions
- Video files (MP4, AVI, etc.) - sent as text descriptions

### Usage
Use the `--media` or `-md` option to specify media files to send with your message:

```bash
# Analyze an image
cliver chat --media path/to/image.jpg "What's in this image?"

# Stream response with media
cliver chat --stream --media path/to/image.jpg "Describe this image in detail"

# Send multiple media files
cliver chat --media path/to/image1.jpg --media path/to/image2.jpg "Compare these images"
```

### Supported LLM Providers
- OpenAI-compatible models (GPT-4 Vision, etc.)
- Ollama models (LLaVA, etc.)

Note: Make sure your LLM model supports multi-media input. Text-only models will not be able to process media files.

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

### LLM Model Configuration
CLIver automatically detects capabilities for different LLM providers. For OpenAI-compatible providers, file upload capability is enabled by default. For other providers, you can explicitly set capabilities in the configuration.

If you're using an OpenAI-compatible provider that doesn't support file uploads, you can configure the model capabilities explicitly in the configuration file to disable the FILE_UPLOAD capability.

For OpenAI-compatible providers that don't support file uploads, CLIver will exit early with a clear error message when you try to use the `--file` option, preventing unnecessary processing.

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