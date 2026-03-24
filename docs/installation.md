---
title: Installation
description: How to install and set up CLIver on your system
---

# Installation Guide

This guide will walk you through installing CLIver on your system, whether you want to use it as a command-line tool or as a Python library in your projects.

## System Requirements

Before installing CLIver, ensure your system meets the following requirements:

- **Python Version**: Python 3.10 or higher
- **Operating System**: Linux

## Quick Installation

The easiest way to install CLIver is using pip:

```bash
pip install cliver
```

This will install CLIver as both a command-line tool and a Python library.

## Verify Installation

After installation, verify that CLIver is properly installed by running:

```bash
cliver --version
```

You should see the installed version number.

## Install from Source

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/cliver-project/CLIver.git
```

Or clone and install:

```bash
git clone https://github.com/cliver-project/CLIver.git
cd cliver
pip install -e .
```

## CLI Quickstart

Once installed, you can start using CLIver immediately:

```bash
# Start a chat session with default settings
cliver

# Check available commands
cliver --help

# Get help for a specific command
cliver chat --help
```

## Docker

CLIver is also available as a Docker image from GitHub Container Registry.

### Pull and run

```bash
# Latest version
docker run --rm -it -v ~/.cliver:/home/cliver/.cliver ghcr.io/cliver-project/cliver chat

# Specific version
docker run --rm -it -v ~/.cliver:/home/cliver/.cliver ghcr.io/cliver-project/cliver:0.0.3 chat
```

### Volume mount

The `-v ~/.cliver:/home/cliver/.cliver` mount maps your local config directory into the container. This persists all CLIver data between runs:

```
~/.cliver/
├── config.yaml              # LLM providers, MCP servers
├── memory.md                # Global memory
└── agents/{agent_name}/
    ├── memory.md            # Agent-specific memory
    ├── identity.md          # Agent identity profile
    ├── sessions/            # Conversation sessions
    └── tasks/               # Scheduled tasks
```

Without the mount, all configuration and memory is lost when the container exits.

### Working with project files

To give CLIver access to your project directory (for file operations, code review, etc.):

```bash
docker run --rm -it \
  -v ~/.cliver:/home/cliver/.cliver \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/cliver-project/cliver chat
```

### Environment variables

Pass API keys and other environment variables with `-e`:

```bash
docker run --rm -it \
  -v ~/.cliver:/home/cliver/.cliver \
  -e OPENAI_API_KEY \
  ghcr.io/cliver-project/cliver chat
```

> **Note:** The container runs as user `cliver` (UID 1001). Files created in mounted volumes will be owned by UID 1001. If this doesn't match your host user, adjust with `--user $(id -u):$(id -g)`.

## Next Steps

After installation, proceed to the [Configuration Guide](configuration.md) to set up your LLM providers and other settings, or check out the [Chat Command Usage](chat.md) to start using CLIver immediately.