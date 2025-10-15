---
title: Session Command
description: Learn how to manage persistent options for interactive chat sessions
---

# Session Command Usage

The `cliver session` command allows you to manage persistent options that will be used across all chat commands during an interactive session. This feature enables you to set configuration options once and have them apply to all subsequent interactions without needing to specify them repeatedly.

## Basic Usage

To view current session options:

```bash
cliver session
```

To list current session options explicitly:

```bash
cliver session --list
```

## Setting Session Options

### Model Selection

Set a specific model to use for all subsequent chat commands:

```bash
cliver session --model gpt-4-turbo
```

### Temperature Control

Set the temperature parameter to control response creativity:

```bash
cliver session --temperature 0.7
```

### Advanced Configuration Options

Set multiple inference parameters at once:

```bash
cliver session \
  --temperature 0.5 \
  --max-tokens 1024 \
  --top-p 0.9 \
  --frequency-penalty 0.3
```

### Enable Streaming

Enable streaming responses for all subsequent chats:

```bash
cliver session --stream
```

### Skill Sets and Templates

Set default skill sets or templates:

```bash
cliver session --skill-set code_review --skill-set documentation
cliver session --template code_review_template
```

### Additional Options

Set additional inference options using key-value pairs:

```bash
cliver session --option presence_penalty=0.6 --option stop="['END']"
```

## Managing Session Options

### Reset All Options

Reset all session options to their defaults:

```bash
cliver session --reset
```

### Clear Specific Options

Clear specific session options without affecting others:

```bash
cliver session --clear --model  # Clears only the model option
cliver session --clear --temperature  # Clears only the temperature option
cliver session --clear --option presence_penalty  # Clears only specific additional option
```

## Using in Interactive Sessions

The session command is especially useful in interactive mode. Here's a typical workflow:

```bash
# Start interactive mode
cliver

# Set up your preferred configuration
Cliver> session --model gpt-4-turbo --temperature 0.5 --stream

# Now all chat commands will use these settings
Cliver> chat "Hello, how are you?"
Cliver> chat "Can you help me with Python?"
Cliver> chat "Explain decorators in Python."

# Check current session options
Cliver> session

# Modify just one setting
Cliver> session --temperature 0.8

# Continue chatting with new settings
Cliver> chat "Now be more creative!"

# Reset when done
Cliver> session --reset
```

## Integration with Chat Command

When using the `cliver chat` command in an interactive session, options specified directly in the command will take precedence over session options:

```bash
# Uses session model, but overrides temperature for this command only
cliver chat --temperature 0.9 "This command uses custom temperature"

# After this command, the session temperature remains unchanged
```

## Examples

### Example 1: Code Review Session
```bash
cliver session \
  --model gpt-4-turbo \
  --temperature 0.2 \
  --skill-set code_review \
  --template code_review_template \
  --stream
```

### Example 2: Creative Writing Session
```bash
cliver session \
  --model claude-3-opus-20240229 \
  --temperature 0.8 \
  --top-p 0.9 \
  --max-tokens 2048
```

### Example 3: Quick Configuration Change
```bash
# Check current settings
cliver session

# Make quick adjustment
cliver session --frequency-penalty 0.5

# Verify change
cliver session --list
```

## Next Steps

Now that you understand session management, you can optimize your interactive workflows by setting up your preferred configurations once and using them consistently throughout your session.