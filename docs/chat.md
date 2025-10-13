---
title: Chat Command
description: Learn how to use the cliver chat command to interact with different LLMs
---

# Chat Command Usage

The `cliver chat` command provides an interactive interface for communicating with various large language models. This guide covers all the features and options available when using the chat functionality.

## Basic Usage

To start a simple chat session with the default model:

```bash
cliver chat
```

This will open an interactive session using your configured default LLM provider.

## Selecting Different Models

### OpenAI Models

To chat with OpenAI's models:

```bash
# Use GPT-4 Turbo
cliver chat --model gpt-4-turbo

# Use GPT-3.5 Turbo
cliver chat --model gpt-3.5-turbo

# Use GPT-4
cliver chat --model gpt-4
```

## Chat Configuration Options

### Temperature Control

Control the creativity of the model's responses:

```bash
# More creative responses (higher temperature)
cliver chat --model gpt-4-turbo --temperature 0.9

# More deterministic responses (lower temperature)
cliver chat --model gpt-4-turbo --temperature 0.2
```

### System Prompt

Set a system prompt to guide the model's behavior:

```bash
cliver chat --system "You are a helpful assistant that responds in a professional manner."
```

### Predefined Context

Start a chat with predefined context:

```bash
# Provide initial context via file
cliver chat --context-file /path/to/context.md

# Provide initial context via stdin
cat /path/to/context.txt | cliver chat --context-stdin
```

## Advanced Chat Features

### Using MCP Servers

Integrate with Model Context Protocol servers for enhanced context:

```bash
# Enable MCP integration (if configured)
cliver chat --mcp-enabled

# Use specific MCP server
cliver chat --mcp-server local-mcp-server
```

### Custom Parameters

Pass custom parameters to the LLM:

```bash
# Set max tokens for response
cliver chat --max-tokens 1024

# Set top_p parameter for nucleus sampling
cliver chat --top-p 0.9

# Set frequency penalty
cliver chat --frequency-penalty 0.5
```

### Multiple Provider Selection

Chat with multiple providers simultaneously:

```bash
# Compare responses from multiple models
cliver chat --compare gpt-4-turbo claude-3-opus-20240229 gemini-pro
```

## File Integration

Work with files directly in the chat:

```bash
# Include a file in your message
cliver chat "Can you summarize this document?" --file /path/to/document.txt

# Process multiple files
cliver chat "Compare these two files" --file /path/to/file1.txt --file /path/to/file2.txt
```

## Batch Mode

For non-interactive use, you can run chat in batch mode:

```bash
# Single query in batch mode
cliver chat --batch --query "What is the capital of France?"

# Process multiple queries from a file
cliver chat --batch --queries-file /path/to/queries.txt
```

## Examples

### Example 1: Professional Assistant Session
```bash
cliver chat \
  --model gpt-4-turbo \
  --system "You are a professional technical assistant. Provide concise, accurate answers with examples when possible." \
  --temperature 0.3
```

### Example 2: Creative Writing Assistant
```bash
cliver chat \
  --model claude-3-opus-20240229 \
  --system "Help me brainstorm creative writing ideas. Be imaginative and provide detailed suggestions." \
  --temperature 0.8
```

### Example 3: Code Review Session
```bash
cliver chat \
  --model gpt-4-turbo \
  --system "Review this code for best practices, security issues, and potential improvements." \
  --file /path/to/code.py
```

## Troubleshooting

### Common Issues

**Issue: API key not found**
- Solution: Ensure your API key is set in the configuration file or as an environment variable

**Issue: Model not available**
- Solution: Check that the model name is correct and available through your provider

**Issue: Request timeout**
- Solution: Check your internet connection or try a different model provider

## Next Steps

After mastering the chat command, learn how to define [Workflows](workflow.md) to automate complex multi-step operations, or check out the [Extensibility Guide](extensibility.md) to learn how to use CLIver as a Python library.