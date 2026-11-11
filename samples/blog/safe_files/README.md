# Ask before the agent touches the filesystem

The agent gets the builtin `Read` and `Write` tools. `Read` is a `SAFE` tool and runs directly;
`Write` is `ASK`, so the agent consults the sample's permission callback first. A second guard —
an `ExecutionPolicy` — limits every file tool to the workspace and writes to `drafts/`.

## Install and run

```bash
pip install cliver-core==0.3.1
export DEEPSEEK_API_KEY=...   # variable name only; never commit a value
python safe_files.py
```

From this repository (workspace install):

```bash
uv run python samples/blog/safe_files/safe_files.py
```

The sample creates a throwaway workspace containing `drafts/plan.md`, asks the model to summarize
it into `summary.md`, and prints the answer plus the files that exist afterwards. The write is
denied with a reason, and the model repeats that reason.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/safe_files -q
```

The tests script the model's tool calls. They cover the callback denial, the path scope denying a
write the callback allowed, and an approved write inside `drafts/` succeeding. Only the live
check calls a real provider.
