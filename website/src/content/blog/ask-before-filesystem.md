---
title: "Ask before the agent touches the filesystem"
description: "An ASK-category tool goes through your callback, a denial is a normal tool result, and path scopes bound what the tool can touch anyway."
pubDate: 2026-11-16
tags:
  - Policy
  - Permissions
  - AgentCore
issue: 5
draft: true
author: Lin Gao
series: Field Notes
---

A useful agent can read files and write them. That is also the moment a demo becomes a
liability: the model chooses which path to touch, and nothing in the loop knows which paths you
care about.

CLIver splits this into two opt-in axes. The **permission axis** decides whether a tool may run;
the **environment axis** decides what a running tool can touch. Both are supplied by the
application — there are no hidden defaults to discover.

## The gate

```python title="safe_files.py" file="samples/blog/safe_files/safe_files.py" region="policy"
```

`Read` is declared `SAFE` in the builtin toolset, so it runs without consulting the callback.
`Write` is `ASK`, so every call goes through `approve()` first. The callback sees the tool name,
its arguments, and the MCP server if there is one.

A denial is not an exception. `Decision.deny(reason)` comes back to the model as a normal tool
result with `denied: true`, and the run continues; a callback that raises fails closed instead.
A broken prompt never widens permissions.

## The second boundary

```python title="safe_files.py" file="samples/blog/safe_files/safe_files.py" region="agent"
```

The `ExecutionPolicy` is ambient while tools execute: `Read` and friends resolve relative paths
against `cwd` and reject reads outside `read_paths`; `Write` rejects writes outside `write_paths`.
Symlinks are followed before comparison, so a link cannot smuggle a path out of scope. The same
policy builds the subprocess environment for `Bash` and `Exec`, and `run_skill()` re-pins `cwd` to
the skill directory so a skill's relative paths keep working.

Keep the layers separate. The callback is where judgment lives — a UI prompt, a tenant rule, an
allowlist. Path scopes are the guardrail for the case where judgment was wrong.

## The run

````text
The write was refused.

**Exact reason returned by the tool:**

```
Error: Permission denied: writes are limited to the drafts/ directory
```

So `drafts/plan.md` reads fine (its content: "Ship the permission post on Wednesday."), but `summary.md` sits outside the only writable directory. I haven't written anything anywhere. If you'd like, I can put the summary at `drafts/summary.md` instead — the one-sentence summary would be: *"The plan is to ship the permission post on Wednesday."*

files: ['drafts/plan.md']
````

The model read `drafts/plan.md`; the attempted `summary.md` write never reached the filesystem.
It also offers `drafts/summary.md` as an alternative — the denial is a conversation, not a dead
end.

## A guardrail, not a sandbox

Path scopes bind the file tools. An approved `Bash` or `Exec` call still runs as your process
user and can touch anything that user can, under the filtered environment and pinned directory.
That is why the permission axis is the one that should decide whether shell access is granted at
all: scopes reduce accidents, they are not OS-level confinement.

The CLI wires the same axis to an interactive prompt; `--yes` approves ASK tools without asking,
and a non-interactive run denies them with a hint instead of hanging. The CLI is experimental and
not on PyPI yet.

Next: [Permissions & Policy](../../core-concepts/policy/) covers both axes, `safety_overrides`,
and the full scope rules.
