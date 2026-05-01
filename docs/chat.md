---
title: Chat
description: Learn how to use CLIver to interact with different LLMs
---

# Chat Usage

CLIver provides an interactive interface for communicating with various large language models. Chat is the default mode — no subcommand needed. This guide covers all the features and options available.

## Basic Usage

To start an interactive session with the default model:

```bash
cliver
```

This will open an interactive session using your configured default LLM provider.

You can also pass a query directly:

```bash
cliver "What is the capital of China?"
```

## Selecting Different Models

### DeepSeek Models

To chat with DeepSeek models:

```bash
# Use DeepSeek-R1, note that you need to configure the model first
cliver --model deepseek-r1
```

### QWen3 Models

To chat with QWen3 coder model:

```bash
# Use Qwen3-Coder, note that you need to configure the model first
cliver --model qwen3-coder
```

## Chat Configuration Options

### Temperature Control and others

Control the creativity of the model's responses:

Without `query` specified, it starts an interactive session with the options specified as the default value.

```bash
# More creative responses (higher temperature)
cliver --model deepseek-r1 --temperature 0.9

# More deterministic responses (lower temperature)
cliver --model deepseek-r1 --temperature 0.2

# Set max tokens for response
cliver --max-tokens 1024

# Set top_p parameter for sampling
cliver --top-p 0.9

# Set frequency penalty
cliver --frequency-penalty 0.5
```

### System Prompt

Set a system prompt to guide the model's behavior:

> NOTE: the system message will be appended to the builtin system message if specified.

```bash
cliver --system-message "You are a helpful assistant that responds in a professional manner."
```

## Advanced Chat Features

### Using MCP Servers

As long as MCP servers are configured, all tools will be included by default.

You can filter the tools using `--included-tools` option:

```bash
cliver --included-tools "*time"
```

### Using Skills

CLIver has an LLM-driven skill system. During a chat session, the LLM can discover and activate skills automatically using the builtin `skill` tool, or you can activate them manually:

```
/skills run brainstorm design a login page
```

Skills are defined as SKILL.md files discovered from `.cliver/skills/` (project), `~/.config/cliver/skills/` (global), and other compatible directories.

See [Skills](skills.md) for details on creating and using skills.

## File Integration

Work with files directly in the chat:

```bash
# Include a file in your message
cliver "Can you summarize this document?" --file /path/to/document.txt

# Process multiple files
cliver "Compare these two files" --file /path/to/file1.txt --file /path/to/file2.txt
```

## Examples

### Example 1: Professional Assistant Session
```bash
cliver \
  --model qwen3-coder \
  --system-message "You are a professional technical assistant. Provide concise, accurate answers with examples when possible." \
  --temperature 0.3
```

### Example 2: Creative Writing Assistant
```bash
cliver \
  --model deepseek-r1 \
  --system-message "Help me brainstorm creative writing ideas. Be imaginative and provide detailed suggestions." \
  --temperature 0.8
```

### Example 3: Code Review Session
```bash
cliver \
  --model qwen3-coder \
  --system-message "Review this code for best practices, security issues, and potential improvements." \
  --file /path/to/code.py
```

## Next Steps

After mastering the chat command, learn about [Skills](skills.md) for specialized task activation, [Memory & Identity](memory-identity.md) for agent personalization, or check out [Workflows](workflow.md) to automate complex multi-step operations.
