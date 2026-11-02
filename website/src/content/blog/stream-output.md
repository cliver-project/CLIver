---
title: "Stream output without changing your code"
description: "The same AgentCore, the same bindings: iterate stream() instead of awaiting chat(), and keep the final message."
pubDate: 2026-11-06
tags:
  - AgentCore
  - Streaming
issue: 2
draft: true
author: Lin Gao
series: Field Notes
---

A batch job can wait for the whole answer. A chat window cannot. The usual workaround is a second
code path for streaming, with its own message assembly and its own bugs.

CLIver does not have a second path. `stream()` runs the same agent loop as `chat()`: same
bindings, same tools, same message objects. The only difference is how you consume the result.

## Deltas as they arrive

```python title="stream_demo.py" file="samples/blog/stream_demo/stream_demo.py" region="stream"
```

`chunk.content` is a text delta, not a growing snapshot. Append it if you also need the final
string — here the sample returns the accumulated answer after printing it.

The loop still assembles the assistant turn from the provider's terminal message, so tool calls
and multi-iteration runs behave the same as in `chat()`. Ordinary chunks carry text; the terminal
one carries the native message the core converts into the next conversation turn.

Cleanup is ordinary async hygiene: `aclose()` runs in `finally`, whether the stream finishes,
raises, or the consumer stops early. Closing the generator or cancelling the task releases the
provider's HTTP stream.

## The run

```text
An agent loop is the repeated cycle in which an AI agent observes the current state, decides on
an action, and uses tools to act on the environment. It then observes the result and repeats this
process until the task is complete or a stopping condition is met.
(260 characters)
```

The printed text and the returned string are the same characters.

## When to use it

- Use `stream()` when a person is watching: chat UIs, terminals, live logs.
- Use `chat()` when the next step needs the finished answer: validation, storage, another call.
- Both accept `conversation=`, `extra_tools=`, and the rest of the `chat()` parameters.

The CLI uses this path directly: `uv run cliver --yes` starts a streaming REPL with multi-turn
history. The CLI is experimental and not on PyPI yet; the repository version runs the same
`AgentCore` you just read.

Next: [Structured Output](../../core-concepts/structured-output/) turns the final answer into a
validated object instead of text.
