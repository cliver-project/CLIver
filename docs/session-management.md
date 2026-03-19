---
title: Session Management
description: Conversation sessions, history, compression, and session options
---

# Session Management

CLIver provides comprehensive session management for conversation history, persistence, and inference options.

## Conversation Sessions

### Session Commands

```bash
# Show current session info
cliver session

# List all saved sessions
cliver session list

# Load a previous session to continue the conversation
cliver session load <session_id>

# Start a new empty session
cliver session new

# Delete a session
cliver session delete <session_id>

# Force-compress current conversation history
cliver session compress
```

### How Sessions Work

- Each interactive chat session is automatically saved as a JSONL file
- Sessions are stored per-agent at `~/.config/cliver/agents/{agent_name}/sessions/`
- Loading a session restores the conversation history, allowing you to continue where you left off
- Session history is converted to LangChain `BaseMessage` objects and passed to the LLM

### Session Permissions

Session-scoped permission grants are in-memory and cleared when the session ends:

```bash
# Show session grants
cliver session permission

# Override mode for this session only
cliver session permission mode yolo

# Allow/deny specific tools this session
cliver session permission grant web_fetch
cliver session permission deny docker_run

# Clear all session grants
cliver session permission clear
```

## Conversation Compression

CLIver automatically compresses conversation history when it approaches the model's context window limit.

### How Compression Works

1. **Trigger**: Compression activates at 70% of the model's context window
2. **LLM-based**: An LLM summarizes older turns into a concise summary
3. **Preservation**: The newest 30% of turns are preserved verbatim
4. **Fallback**: If LLM compression fails, a truncation fallback removes the oldest turns
5. **Token estimation**: Uses `len(text) // 4` heuristic (no external tokenizer dependency)

### Manual Compression

Force-compress the current conversation at any time:

```bash
cliver session compress
```

### Context Window Configuration

Each model can specify its context window size:

```yaml
models:
  qwen:
    name_in_provider: "qwen2.5:latest"
    provider: ollama
    url: "http://localhost:11434"
    context_window: 32768
```

If not specified, CLIver uses heuristic defaults based on the model name.

## Session Options

The `session-option` command manages persistent inference options that apply to all subsequent chat commands during an interactive session.

### Viewing Options

```bash
cliver session-option
```

### Setting Options

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

### Resetting Options

```bash
cliver session-option reset
```

### Interactive Session Workflow

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

## Token Usage Statistics

Track token consumption across sessions:

```bash
# Show current session token usage by model
cliver cost

# Show aggregated usage from audit logs
cliver cost total

# Filter by model or date range
cliver cost total --model qwen --from 2025-01-01 --to 2025-01-31

# Filter by agent
cliver cost total --agent CodeHelper
```
