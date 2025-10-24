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

## Next Steps

After installation, proceed to the [Configuration Guide](configuration.md) to set up your LLM providers and other settings, or check out the [Chat Command Usage](chat.md) to start using CLIver immediately.