# A read-only code review agent

The agent gets one custom tool that can list and read files in a fixture repository. It has no
write tool, and an `ExecutionPolicy` pins the run to the fixture with reads inside it and writes
nowhere. The review comes back through `chat_structured()` as a validated `ReviewReport`.

The fixture (`fixture/`) contains two small files with deliberate problems: a mutable default
argument in `aggregate.py`, and an off-by-one loop plus a bare `except` in `loaders.py`. Findings
are the model's judgments, not a linter's — the sample demonstrates the wiring, not review quality.

## Install and run

```bash
pip install cliver-core==0.3.2
export DEEPSEEK_API_KEY=...   # variable name only; never commit a value
python code_review.py
```

From this repository (workspace install):

```bash
uv run python samples/blog/code_review/code_review.py
```

The sample lists the fixture files, reads them through `InspectRepo`, and prints the summary plus
one `file:line: severity: message` line per finding.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/code_review -q
```

The tests script the model's tool calls and final report. They cover the list-then-read flow, the
validated report, the advertised tool set (only `InspectRepo`), and a read outside the policy
being refused. Only the live check calls a real provider.
