---
title: "The smallest useful agent is a loop and a tool"
description: "A working CLIver agent in about twenty lines — and the three boundaries that keep it from becoming a demo that only works once."
pubDate: 2026-11-02
tags:
  - AgentCore
  - Quickstart
issue: 1
featured: true
draft: false
author: Lin Gao
series: Field Notes
---

Most agent frameworks ask you to learn their graph, their state object, and their deployment
story before you can print a file listing. CLIver starts from the other direction: one async
loop, typed messages, and a tool registry you can read in a single sitting.

Here is the agent itself.

## A working agent

```python title="quickstart.py" file="samples/blog/quickstart/quickstart.py" region="agent"
```

The runnable file adds the imports and an entry point. `presets.deepseek()` reads
`DEEPSEEK_API_KEY` at call time, and the sample's test injects a scripted agent, so the test
never reads a key or touches the network.

That is a real agent loop. The model reads the tool catalogue, chooses a tool, the core runs
it, and the result goes back to the model until it has an answer. No graph definition, no
hidden agent executor.

### What the core owns

CLIver keeps the loop and the wire protocols in `cliver-core`:

- **The loop.** Tool calls, tool results, iteration limits, and the final response.
- **The wire.** OpenAI and Anthropic protocol providers, with provider-specific behavior kept
  in the provider that owns it.
- **The contract.** Message and response objects that an application can persist, stream, or
  render without importing a UI framework.

Everything else — your routes, your database, your tenant model — stays in your application.

## Three boundaries worth keeping

1. **The agent is stateless.** Conversation history is passed per call as `conversation=`.
   One instance can serve concurrent chats inside one event loop.
2. **Tools are explicit.** `builtin_tools` is an allowlist, not a suggestion. A tool the model
   cannot see is a tool it cannot call.
3. **Lifecycle is yours.** `await agent.aclose()` closes the providers the agent built. The core
   does not own your process, your scheduler, or your shutdown policy.

> The useful part of an agent framework is not the magic. It is the set of edges it makes
> explicit before you need them.

## Next

The next note streams the same loop chunk by chunk, without a second code path.
