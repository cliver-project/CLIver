---
title: Extensibility
description: How to extend CLIver functionality and use it as a Python library
---

# Extensibility Guide

CLIver is designed with extensibility in mind, allowing you to customize its functionality, add new features, and integrate it seamlessly into your own Python applications. This guide covers how to extend CLIver and use it as a Python library.

## Using CLIver as a Python Library

CLIver provides a rich Python API that allows you to integrate LLM functionality directly into your applications.

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