---
title: Extensibility
description: How to extend CLIver functionality and use it as a Python library
---

# Extensibility Guide

CLIver is designed with extensibility in mind, allowing you to customize its functionality, add new features, and integrate it seamlessly into your own Python applications. This guide covers how to extend CLIver and use it as a Python library.

## Using CLIver as a Python Library

CLIver's core engine (`TaskExecutor`) is designed to be **independent of the CLI layer**. It has no dependencies on terminal I/O, prompt_toolkit, or Rich — making it suitable for embedding in web services, automation scripts, or other applications.

All features — LLM inference, tool calling, permissions, workflows, skills, and memory — work identically whether invoked from the CLI or from your own Python code.

### Basic LLM Inference

```python
--8<-- "examples/simple_example.py"
```

### Streaming Responses

For real-time applications, you can stream responses from the LLM:

```python
--8<-- "examples/simple_example_stream.py"
```

## Custom Commands

You can add custom commands to CLIver's CLI interface by creating command modules.

### Creating a Custom Command

Create a Python file in the default config location, like: `~/.config/cliver/commands/my_command.py`

```python
--8<-- "examples/my_command.py"
```
The command will be loaded automatically by CLIver

## Next Steps

Now that you understand how to extend CLIver, see our [Roadmap](roadmap.md) for upcoming features and how you can contribute to the project.