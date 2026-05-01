---
title: Session Options
description: Manage persistent inference options for interactive chat sessions
---

# Session Options

The `cliver session-option` command manages persistent inference options that apply to all subsequent chat commands during an interactive session.

## Viewing Options

Run with no arguments to see current settings:

```bash
cliver session-option
```

## Setting Options

Use the `set` subcommand to configure options:

```bash
# Set model
cliver session-option set --model qwen

# Set multiple options at once
cliver session-option set \
  --temperature 0.5 \
  --max-tokens 1024 \
  --top-p 0.9 \
  --stream

# Set additional inference parameters
cliver session-option set --option presence_penalty=0.6
```

### Available Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | LLM model to use |
| `--temperature` | | Temperature parameter |
| `--max-tokens` | | Maximum tokens |
| `--top-p` | | Top-p parameter |
| `--frequency-penalty` | | Frequency penalty |
| `--template` | `-t` | Prompt template |
| `--stream` | `-s` | Enable streaming |
| `--no-stream` | | Disable streaming |
| `--save-media` | `-sm` | Enable media saving |
| `--no-save-media` | | Disable media saving |
| `--media-dir` | `-md` | Media save directory |
| `--included-tools` | | Tool filter pattern |
| `--option` | | Additional key=value options |

## Resetting Options

Reset all options to defaults:

```bash
cliver session-option reset
```

## Interactive Session Workflow

```bash
# Start interactive mode
cliver

# Configure your session
CLIver> session-option set --model qwen --temperature 0.5 --stream

# Chat with those settings
CLIver> chat "Hello, how are you?"
CLIver> chat "Explain decorators in Python."

# Check current settings
CLIver> session-option

# Adjust one setting
CLIver> session-option set --temperature 0.8

# Reset when done
CLIver> session-option reset
```

## Integration with CLI Options

Options specified directly on the `cliver` command line override session options for that invocation only:

```bash
# Uses session model, but overrides temperature for this invocation only
cliver --temperature 0.9 "Be more creative"

# Session temperature remains unchanged after this command
```
