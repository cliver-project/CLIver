---
title: "Test an agent without burning tokens"
description: "Script the provider, assert what the model saw and what the agent did, and run the whole tool loop offline — no key, no tokens, no flakiness."
pubDate: 2026-11-26
tags:
  - Testing
  - AgentCore
  - Tools
issue: 8
draft: false
author: Lin Gao
series: Field Notes
---

An agent test that calls a real model is an integration test with a random variable in the
middle. It costs tokens, it takes seconds or minutes, and it fails for reasons that have nothing
to do with your change. The parts worth pinning — did the tool result reach the model? did the
permission gate run? what did the run cost? — are deterministic.

This sample builds a tiny agent with one tool and tests two things offline: a denied tool call
reaches the model as a tool result, and `usage_total` covers every model call in the run.

## The agent under test

One tool, and a permission callback that refuses it:

```python title="agent.py" file="samples/blog/test_agent/agent.py" region="tool"
```

```python title="agent.py" file="samples/blog/test_agent/agent.py" region="agent"
```

`SaveNote` is a normal `@tool` function: the schema comes from the type hints. It is `mutating`
by default, so the agent asks the permission callback before running it. The callback here says
no — which is exactly the path we want to test.

## The seam

`AgentCore` reaches a model through a `Provider`, and the provider arrives through the binding:

```python title="_shared/testing.py" file="samples/_shared/testing.py" region="scripted-provider"
```

The shared helper in `samples/_shared/testing.py` swaps the wire client for a list of prepared
responses. It records every `CLIverRequest` the loop makes, so a test can assert on the messages
the model actually saw. A script entry can be a plain string (a final answer), a full
`CLIverResponse` (including tool calls and usage), an exception (to test failure paths), or a
`Streamed` set of chunks. The helper lives in the sample harness, not in `cliver-core` — it is a
pattern to copy, not a public mock library.

## A denial is a tool result

```python title="test_agent.py" file="samples/blog/test_agent/test_agent.py" region="denial"
```

The first scripted response asks for `SaveNote`; the agent denies it before the body runs. The
second scripted response is the model's answer to the tool result it just received. The
assertions pin three contracts: the file was never written, the tool message says
`Error: Permission denied: ...`, and the event stream is `START`, `PERMISSION_DENIED`, `END`.

That last one matters: a denial that silently disappears or one that runs the tool anyway both
pass a test that only checks the final answer. Events are core behavior, so pin them.

## Usage is a run, not a call

```python title="test_agent.py" file="samples/blog/test_agent/test_agent.py" region="usage"
```

The run makes two model calls: one asks for the tool (10 input + 2 output tokens), one answers
(20 + 5). `response.usage` is the final call; `response.usage_total` is the sum across the run:
30 input and 7 output tokens. The core does that merge, so a test can pin it without a provider
reporting anything special.

## Run it

```bash
cd samples && uv run pytest blog/test_agent -q
```

```text
2 passed in 1.33s
```

No API key, no network, no model decisions. The same test runs in CI on every commit.

## What this does not test

It does not tell you whether the model picks the right tool, or whether a provider's wire format
still works. Those are integration questions. Keep a small, gated live suite for them and do not
make the offline suite depend on it. The shared helper supports streamed scripts too, so
streaming behavior can be pinned the same way.

## In the CLI

The `cliver` CLI runs the same `AgentCore`, but it is for using an agent, not testing one: there
is no CLI test mode, and a CLI invocation is an integration run against a real provider. Keep
this pattern in pytest against the library. The repo's own CLI tests script the provider the same
way; live provider checks are a separate, gated suite.

Next: [API reference](../../api/) covers `AgentCore.chat()`, the response fields including
`usage` and `usage_total`, and the event types.
