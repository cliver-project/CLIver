---
title: "Your first tool is just a Python function"
description: "One decorator, type hints in, JSON Schema out — then watch the model call your function and use the result."
pubDate: 2026-11-09
tags:
  - Tools
  - AgentCore
issue: 4
draft: true
author: Lin Gao
series: Field Notes
---

A tool is the agent's way to do something the model cannot do alone: read a file, query a
database, call your API. In CLIver, a tool is a typed Python function with a decorator.

## The function

```python title="word_count_tool.py" file="samples/blog/word_count_tool/word_count_tool.py" region="tool"
```

`@tool` reads the signature and builds the JSON Schema: `text` is a string, and it is required
because it has no default. The first line of the docstring becomes the model-facing description.
The function returns `list[dict]`; an entry with `"text"` is rendered as the tool result, and an
entry with `"error"` is rendered as an error.

## Wiring it in

```python title="word_count_tool.py" file="samples/blog/word_count_tool/word_count_tool.py" region="agent"
```

`builtin_tools` is an allowlist. The tool you pass is the tool the model can see and the tool the
core will execute — a call to anything else is rejected before the permission check. The tool's
default safety category is `mutating`, so an agent with a permission callback asks before running
it; a read-only function can opt into `safety="safe"`.

## The run

```text
The text 'Agents are loops with tools' contains **5 words**.
```

The model called `word_count` with the sentence, the core appended the tool result (`5`) to the
conversation, and the model wrote the final answer from it. Tool events expose the same steps:
`tool.start` and `tool.end` carry the tool name, arguments, duration, and the run id when you
pass `on_event=`.

## Scope, not suggestion

The advertised set is the execution set. Remove a tool with `tool_filter=`, restrict it with a
skill's `allowed-tools`, or pass an explicit `mcp_servers=` list, and the model's call is rejected
rather than silently run. That boundary is worth keeping: a tool the model cannot see is a tool
it cannot misuse.

The CLI ships a builtin toolset on the same execution path. Custom tools are library-level today;
the sample runs the same core the CLI does. The CLI is experimental and not on PyPI yet.

Next: [Tools](../../core-concepts/tools/) covers schema generation, async tools, tool filters,
and parallel calls.
