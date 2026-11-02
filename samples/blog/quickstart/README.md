# Quickstart — your first CLIver agent

The smallest useful `cliver-core` agent: bind a chat model, advertise the builtin
tools, call `chat()`, and print the answer and token usage. The model decides
whether to list the directory before answering.

## Install and run

```bash
pip install cliver-core==0.3.0
export DEEPSEEK_API_KEY=...   # variable name only; never commit a value
python quickstart.py
```

From this repository (workspace install):

```bash
uv run python samples/blog/quickstart/quickstart.py
```

Expected output: the model's answer, then a `tokens: N` line.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/quickstart -q
```

The test injects a scripted provider, so it never reads `DEEPSEEK_API_KEY` and
never calls a model endpoint.
