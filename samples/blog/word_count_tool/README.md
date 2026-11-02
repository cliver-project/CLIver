# Your first tool is just a Python function

`@tool` turns a typed function into a `CLIverTool`: the JSON Schema comes from the type hints,
the first docstring line becomes the model-facing description, and the agent calls the function
when the model asks for it.

## Install and run

```bash
pip install cliver-core==0.3.0
export DEEPSEEK_API_KEY=...   # variable name only; never commit a value
python word_count_tool.py
```

From this repository (workspace install):

```bash
uv run python samples/blog/word_count_tool/word_count_tool.py
```

Expected output: a one-line answer. The model calls `word_count` first; pass `on_event=` to the
agent to log `tool.start` / `tool.end` events.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/word_count_tool -q
```

The test scripts a tool call, then a final answer, and asserts that the tool is advertised, runs,
and its result reaches the second model request.
