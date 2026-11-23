---
title: "A read-only code review agent"
description: "One read-only tool, a typed report schema, and a path policy let an agent inspect a repository and return findings it cannot act on."
pubDate: 2026-11-24
tags:
  - Tools
  - Structured Output
  - Policy
issue: 7
draft: false
author: Lin Gao
series: Field Notes
---

A code review agent has an unusual risk profile. A bad summary wastes your time; an agent that
decides to fix the code wastes your afternoon. The useful shape is read-only: it can inspect a
repository, and it cannot write to it.

This sample composes the previous three posts: a custom tool, a Pydantic schema through
`chat_structured()`, and an `ExecutionPolicy` that bounds the run.

## The repository

The fixture is two small modules with three deliberate problems: a mutable default argument, an
off-by-one loop, and a bare `except`.

```python title="fixture/aggregate.py"
"""Aggregate scores for the weekly report."""


def average(scores):
    return sum(scores) / len(scores)


def top_scores(scores, limit=3, cache={}):
    key = tuple(scores)
    if key not in cache:
        cache[key] = sorted(scores, reverse=True)[:limit]
    return cache[key]
```

```python title="fixture/loaders.py"
"""Load score files for the weekly report."""


def read_scores(paths):
    scores = []
    for index in range(len(paths) - 1):
        try:
            with open(paths[index], encoding="utf-8") as handle:
                scores.append(int(handle.read()))
        except:
            scores.append(0)
    return scores
```

The goal is not to catch these specific bugs — it is to let the model look around and report what
it sees, with no write path in the loop.

## One read-only tool

```python title="code_review.py" file="samples/blog/code_review/code_review.py" region="tool"
```

`InspectRepo` lists the Python files when `path` is empty; otherwise it returns one file with line
numbers. `safety="safe"` means it runs without a permission prompt.

The read-only part is not a prompt instruction. The tool resolves the path against the policy's
pinned working directory, calls `scope_violation()` from `cliver_core.policy`, and returns an
error to the model when the target falls outside the configured read roots. A relative `../` path
is refused before any file is opened.

That call is deliberate: `cliver-core` applies the execution policy to the builtin file tools, but
a custom tool body is your code. If it does not consult `pinned_cwd()` and `scope_violation()`,
the policy does not apply to it. The helpers are public so a project tool can honor the same
contract.

## The whole advertised surface

```python title="code_review.py" file="samples/blog/code_review/code_review.py" region="agent"
```

`builtin_tools=[inspect_repo]` is the entire toolset. There is no `Write`, no `Bash`, no `Exec`,
and no builtin `Read` — just this one function. AgentCore advertises exactly the tools it will
execute, so a tool the model cannot see is a tool it cannot call. The read-only guarantee is
structural, not behavioral.

The policy is the second layer:

```python title="code_review.py" file="samples/blog/code_review/code_review.py" region="policy"
```

`cwd` pins relative paths to the fixture, `read_paths=[root]` bounds reads, and `write_paths=[]`
says no path is writable. That last field is a guard for any policy-aware tool you might add
later; today there is no write tool to constrain.

## Findings as a typed object

```python title="code_review.py" file="samples/blog/code_review/code_review.py" region="schema"
```

`Finding` carries the file, a 1-based line, a severity, and a message. `ReviewReport` carries a
summary and the list. The model returns one JSON object; `chat_structured()` validates it and, if
the shape is wrong, retries with a bounded error diff — the machinery from post 3.

```python title="code_review.py" file="samples/blog/code_review/code_review.py" region="review"
```

The policy rides the `chat_structured()` call, not the agent, so the same agent can review a
different root without rebuilding. `result.value` is a `ReviewReport`; `attempts` and
`usage_total` account for every model call in the run.

## The run

````text
Two small modules with several real correctness bugs: loaders.read_scores drops the last path
due to an off-by-one range and swallows all errors with a bare except, while aggregate.average
crashes on empty input and aggregate.top_scores uses a mutable default cache keyed without
`limit`, producing stale/incorrect results across calls. Nothing here is covered by error
signalling, so failures degrade silently into zeros or stale data.
loaders.py:6: high: `range(len(paths) - 1)` iterates only up to the second-to-last index, so the final path is never read and its score is silently omitted from the report; use `for path in paths` or `range(len(paths))`.
aggregate.py:8: high: Mutable default argument `cache={}` is shared across all calls, so cached results leak between unrelated invocations and grow without bound (a memory leak). Use `cache=None` with an internal dict, or a proper lru_cache.
aggregate.py:9: high: The cache key is only `tuple(scores)` and ignores `limit`, so calling `top_scores(scores, limit=5)` after `top_scores(scores)` returns the stale 3-element list. Include `limit` in the key.
aggregate.py:5: medium: `sum(scores) / len(scores)` raises ZeroDivisionError for an empty sequence, so an empty (or fully failed) score file set crashes the weekly report instead of returning a defined value such as 0.0.
loaders.py:10: medium: Bare `except:` catches everything including KeyboardInterrupt/SystemExit and hides genuine bugs (missing file, permission error, non-numeric content) by substituting 0, so corrupt data is indistinguishable from a real zero score. Catch `(OSError, ValueError)` and log or re-raise.
loaders.py:11: medium: Substituting 0 on failure keeps the output list misaligned with its input: callers cannot tell which path failed, and the length also differs because of the off-by-one, so downstream index-based lookups pair scores with the wrong files.
loaders.py:9: low: `int(handle.read())` assumes the whole file is a single integer; a trailing blank line or multiple values raises ValueError, which is then silently swallowed into a 0 score.
aggregate.py:11: low: Values are cached as a mutable list returned directly to callers; any caller mutating the result corrupts the cache for all later calls. Return a copy (e.g., `list(...)`) or store a tuple.
attempts: 1
````

All three seeded issues are there — the shared default at `aggregate.py:8`, the skipped last path
at `loaders.py:6`, the bare `except` at `loaders.py:10` — plus five more the model noticed. It
took one attempt.

## What this is not

- **Not a linter.** A linter has rules, stable IDs, and a fixed cost. These are model judgments.
  The severity labels are the model's, not a calibrated scale. Run Ruff or your type checker for
  rules you need to enforce; use the agent for the parts that need reading.
- **Not a sandbox.** The path policy bounds tools that honor it; a tool that ignores
  `scope_violation()` is outside the contract, and an approved shell tool would run with your
  process user's permissions. Post 5 covers that boundary.
- **Not a guarantee about line numbers.** The sample's tool returns numbered lines, which is why
  the findings point at the right rows. If your tool returns raw text, the model has to count,
  and those counts drift.

## In the CLI

The CLI is on PyPI as `cliver==0.3.2` (`uv run cliver` in a checkout). The same read-only shape
works on a diff:

```bash
pip install cliver==0.3.2
git diff | cliver "Review this diff for correctness problems."
```

Leave `--yes` off. In a non-interactive run, the CLI denies every ASK tool — `Write`, `Bash`,
`Exec`, `WebFetch` — with a reason, while `Read`, `LS`, and `Grep` run directly. The model can
inspect the repository and the diff; it cannot change either. This one-liner has no path scope,
so the permission gate is the boundary; post 5 shows how to add the other layer.

Next: [Permissions & Policy](../../core-concepts/policy/) covers the ambient scopes and the
permission axis; [Structured Output](../../core-concepts/structured-output/) covers the schema
contract and the retry behavior.
