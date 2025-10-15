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

### DeepSeek Models

To chat with DeepSeek models:

```bash
# Use DeepSeek-R1, note that you need to configure the model first
cliver chat --model deepseek-r1
```

### QWen3 Models

To chat with QWen3 coder model:

```bash
# Use Qwen3-Coder, note that you need to configure the model first
cliver chat --model qwen3-coder
```

## Chat Configuration Options

### Temperature Control and others

Control the creativity of the model's responses:

Without `query` specified, it starts an interactive session with the options specified as the default value.

```bash
# More creative responses (higher temperature)
cliver chat --model deepseek-r1 --temperature 0.9

# More deterministic responses (lower temperature)
cliver chat --model deepseek-r1 --temperature 0.2

# Set max tokens for response
cliver chat --max-tokens 1024

# Set top_p parameter for nucleus sampling
cliver chat --top-p 0.9

# Set frequency penalty
cliver chat --frequency-penalty 0.5
```

### System Prompt

Set a system prompt to guide the model's behavior:

```bash
cliver chat --system-message "You are a helpful assistant that responds in a professional manner."
```

## Advanced Chat Features

### Using MCP Servers

As long as MCP servers are configured, all tools will be included by default.

You can filter the tools using `--included-tools` option:

```bash
cliver chat --included-tools "*time"
```

## File Integration

Work with files directly in the chat:

```bash
# Include a file in your message
cliver chat "Can you summarize this document?" --file /path/to/document.txt

# Process multiple files
cliver chat "Compare these two files" --file /path/to/file1.txt --file /path/to/file2.txt
```

## Examples

### Example 1: Professional Assistant Session
```bash
cliver chat \
  --model qwen3-coder \
  --system-message "You are a professional technical assistant. Provide concise, accurate answers with examples when possible." \
  --temperature 0.3
```

### Example 2: Creative Writing Assistant
```bash
cliver chat \
  --model deepseek-r1 \
  --system "Help me brainstorm creative writing ideas. Be imaginative and provide detailed suggestions." \
  --temperature 0.8
```

### Example 3: Code Review Session
```bash
cliver chat \
  --model qwen3-coder \
  --system "Review this code for best practices, security issues, and potential improvements." \
  --file /path/to/code.py
```

## Next Steps

After mastering the chat command, learn how to define [Workflows](workflow.md) to automate complex multi-step operations, or check out the [Extensibility Guide](extensibility.md) to learn how to use CLIver as a Python library.