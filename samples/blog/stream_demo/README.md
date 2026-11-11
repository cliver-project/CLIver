# Stream output without changing your code

`agent.stream()` yields the answer as it arrives. The sample prints each chunk, accumulates the
same chunks into the final string, and closes the agent with `aclose()` whether the stream
finishes or raises.

## Install and run

```bash
pip install cliver-core==0.3.1
export DEEPSEEK_API_KEY=...   # variable name only; never commit a value
python stream_demo.py
```

From this repository (workspace install):

```bash
uv run python samples/blog/stream_demo/stream_demo.py
```

Expected output: the answer printed progressively, then its character count. Ctrl-C stops the
demo; the async generator unwinds and the provider stream is released.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/stream_demo -q
```

The test feeds three scripted chunks through the real `AgentCore.stream()` loop.
