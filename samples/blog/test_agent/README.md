# Test an agent without burning tokens

Script the provider and test the whole agent loop offline: no API key, no tokens, no flakiness.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/test_agent -q
```

`agent.py` is the code under test: one `SaveNote` tool behind a denying permission callback.
`test_agent.py` supplies prepared responses through the shared `ScriptedProvider` and asserts
the tool result the model saw, the tool-event sequence, and the usage totals.

The scripted provider lives in `samples/_shared/testing.py`. It is a sample harness, not part of
`cliver-core`.
