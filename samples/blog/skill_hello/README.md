# Load an Agent Skill in one call

An Agent Skill is a directory with a `SKILL.md`: a name, a description, and instructions the
model should follow for a specific task. `cliver-core` never guesses where skills live — the
application supplies the sources — and `run_skill()` pins one skill's instructions into the run.

## Install and run

```bash
pip install cliver-core==0.3.1
export DEEPSEEK_API_KEY=...   # variable name only; never commit a value
python skill_hello.py
```

From this repository (workspace install):

```bash
uv run python samples/blog/skill_hello/skill_hello.py
```

The sample loads `skills/hello`, runs it with the prompt “Greet me in Japanese.”, and prints the
greeting. The skill declares `allowed-tools: Read`, so during the run the agent advertises only
`Read` — the `Write` tool registered on the agent is not available. The skill reads its bundled
`references/phrases.md`; the agent-level `ExecutionPolicy` makes `run_skill()` pin the working
directory to the skill's own directory first.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/skill_hello -q
```

The tests script a `Read` call followed by a final answer and assert that the skill body reached
the system message, that only `Read` was advertised, and that a scripted `Write` call is rejected
before it executes.
