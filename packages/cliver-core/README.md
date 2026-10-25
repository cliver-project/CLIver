# cliver-core

The embeddable agent core behind [CLIver](https://github.com/cliver-project/CLIver): a Re-Act loop
with tool calling, per-purpose model bindings, media capabilities, and first-class MCP support.

No CLI, no database, no config files — you pass Python objects in and get messages back.

## Install

```bash
pip install cliver-core          # core library + MCP client (stdio / streamable HTTP)
pip install "cliver-core[local]" # + in-process local embedding / rerank models
```

Python 3.10+.

## Quick start

```python
import asyncio

from cliver_core import AgentCore, ModelBinding, ModelBindings


async def main():
    agent = AgentCore(
        ModelBindings(
            llm=ModelBinding.llm(
                model="deepseek-v4-flash",
                api_key="...",
                base_url="https://api.deepseek.com",
            )
        )
    )
    response = await agent.chat("Summarize the Re-Act pattern in one sentence.")
    print(response.message.content)
    await agent.aclose()


asyncio.run(main())
```

Ready-made bindings for live-verified providers live in `cliver_core.presets`:

```python
from cliver_core import ModelBindings, presets

bindings = ModelBindings(llm=presets.deepseek(api_key="..."))
agent = AgentCore(bindings)
```

## Tools

`cliver_core.tools` ships nine file, shell, todo, and web tools. Register them directly:

```python
from cliver_core.tools import discover_builtin_tools

agent = AgentCore(bindings, builtin_tools=discover_builtin_tools())
```

`ToolRegistry` and the `@tool` decorator build custom tools.

## Capabilities

Every purpose works standalone — no AgentCore required:

```python
from cliver_core import ModelBinding
from cliver_core.capabilities import embed, generate_image

vectors = await embed(
    texts=["hello", "world"],
    binding=ModelBinding.embed(model="text-embedding-3-small", api_key="...", base_url="..."),
)

image = await generate_image(
    prompt="a red panda on a skateboard",
    binding=ModelBinding.image(model="gpt-image-1", api_key="...", base_url="..."),
)
```

Purposes: image, video, text-to-speech, speech-to-text, embed, rerank. Long generations can return
a `MediaJob` future via `AgentCore`'s `submit_*` variants.

## Local models

Run embedding and rerank models in-process — no endpoint. The backends pull torch, so they are
opt-in:

```bash
pip install "cliver-core[local-st]"     # sentence-transformers embeddings
pip install "cliver-core[local-flag]"   # FlagEmbedding embeddings + rerank
pip install "cliver-core[local]"        # both
```

```python
from cliver_core.capabilities import embed, rerank
from cliver_core.local import (
    LocalEmbeddingProvider,
    LocalRerankProvider,
    local_embed,
    local_rerank,
)

embedder = LocalEmbeddingProvider(
    model="BAAI/bge-base-zh-v1.5",  # Hugging Face id or local directory
    backend="sentence-transformers",  # required; never inferred from the model name
)
reranker = LocalRerankProvider(
    model="BAAI/bge-reranker-v2-m3",
    backend="flag-embedding",
)

vectors = await embed(
    texts=["hello", "world"],
    binding=local_embed(provider=embedder, normalize=True),
)
items = await rerank(
    query="hello",
    documents=["hi", "hey"],
    binding=local_rerank(provider=reranker, top_k=1),
)

await embedder.close()
await reranker.close()
```

A provider owns one loaded model and is meant to be long-lived: create it once, share it across
bindings and calls, and close it at shutdown. There is no process-global model cache.
`ModelBinding.aclose()` never closes a provider you supplied.

The API is deliberately explicit: `model`, `backend`, `device`, and `dtype` are parameters, not
name-based defaults. Call options are generic — `dim` (optional output-size check), `normalize`
(off by default; raw vectors/scores), `batch_size`, and `top_k` (rerank). Query/document prefixes
and instruction templates belong to the application.

`provider.info()` reports the resolved model directory, revision, device, dtype, and load time.
Hugging Face downloads honor `HF_TOKEN`, `cache_dir`, `revision`, and `local_files_only`.

## MCP

```python
from cliver_core.mcp import MCPClient

client = MCPClient.from_discovery()  # ~/.cliver/mcp.json, .cliver/mcp.json, drop-ins
agent = AgentCore(bindings, mcp_client=client)
```

MCP tools are advertised to the model as `mcp__<server>__<tool>`. Prompts and resources are read on
demand — `list_prompts`, `get_prompt`, `list_resources`, `read_resource` — never registered as
tools.

## Skills

[Agent Skills](https://agentskills.io/specification): a skill is a directory with a `SKILL.md`
(frontmatter + instructions) plus optional `scripts/`, `references/`, `assets/`. cliver-core reads
skills from **explicit sources the application supplies** — a directory of skill directories or a
single skill directory; nothing is discovered implicitly, and later sources override earlier ones
by name.

```python
from cliver_core import AgentCore, SkillManager

skills = SkillManager(sources=["./skills"])  # routing metadata: skills.list_skills()
agent = AgentCore(bindings, builtin_tools=[...], skill_manager=skills)

result = await agent.run_skill("Refund", "订单 12345 想退货", conversation=history)
```

`run_skill` (name or an already-loaded `Skill` object) appends an activation header — skill
name and location, so bundled `scripts/` are runnable — plus the skill body to the single
system message and runs the normal Re-Act loop; a declared `allowed-tools` frontmatter narrows
the advertised tools to that set — a non-empty allowlist matching zero available tools fails
fast, while an absent or empty list leaves tools unchanged. The other call parameters mirror
`chat()` (except `tool_filter` and `skill_dispatch`). `run_skill()` adds none of the
model-driven skill tools (`FindSkills`, `ActivateSkill`, `ReadSkillFile`); an `allowed-tools`
entry naming one of those is unavailable in this path.

Progressive disclosure: `list_skills()` exposes only routing metadata (name + description —
`format_skill_catalog()` renders the catalogue for prompts); the body enters context only when a
skill is activated; bundled files are listed (`skill.files`) and readable on demand
(`skill.read_file("references/policy.md")`, path escapes rejected).

With a `SkillManager` configured, `chat()`/`stream()` give the model that catalogue in the system
prompt (`skill_summary_limit`, default 10), `FindSkills(query, limit)` to search the full
catalogue, and `ActivateSkill(name)` to load a skill's body as a tool result for the rest of the
run. Activation is one skill at a time; a declared `allowed-tools` narrows the advertised tools
for the remaining iterations, and an activated skill that bundles files also gets
`ReadSkillFile(path)` — confined to that skill's directory (escapes refused, UTF-8 text, 256 KiB
cap), while workspace files stay on `Read`. Apps may rank the catalogue with
`skill_selector(skills, user_input)`, replace lexical search with
`skill_search(query, skills)`, and narrow both candidate pools with
`candidate_selector(tools, skills, user_input)` / `tool_selector(tools, user_input)`.
`run_skill()` disables dispatch for its run.

The dependency-free frontmatter parser covers the spec's surface plus what cross-agent skills
(workbuddy, accio, ...) carry — flat scalars, quoted strings, `>` / `|` blocks, and nested maps
of any depth (unknown keys are kept in `skill.frontmatter` for forward compatibility; no anchors
or `-` sequences). Validation follows the agentskills.io rules (name
pattern, 64/1024/500-char limits) as load-time warnings; skills stay loadable unless `name` or
the frontmatter itself is missing. (Anthropic's product-specific extras — reserved words in
names — are not enforced.) Script execution sandboxes remain future work.

## Permissions & execution policy

Two independent axes, both opt-in — with neither set, every tool runs exactly as before.

**Permission axis (may this tool run?).** Every tool carries a category: `SAFE` or `ASK`.
Read-only builtins (`Read`, `LS`, `Grep`, `TodoRead`/`TodoWrite`) are `SAFE`; mutating ones
(`Write`, `Bash`, `Exec`, `WebFetch`) are `ASK` by default. MCP tools map the spec's
self-declared `readOnlyHint` annotation to `SAFE` — a hint, never a permission boundary.
An application-supplied async `permission_callback` is consulted before any `ASK` tool runs:

```python
from cliver_core import AgentCore, Decision, EnvSpec, ExecutionPolicy, SAFE_ENV_VARS, ToolCategory


async def ask_user(ctx) -> Decision:  # ctx: tool_name, args, server, category
    answer = await my_ui.confirm(f"Run {ctx.tool_name}?")
    return Decision.allow() if answer else Decision.deny("user said no")


agent = AgentCore(
    bindings,
    builtin_tools=[...],
    permission_callback=ask_user,  # None → ASK tools run unchecked
    execution_policy=ExecutionPolicy(  # environment axis, ambient per call
        env=EnvSpec(mode="allowlist", names=set(SAFE_ENV_VARS)),
        cwd=project_root,
        write_paths=[project_root / "src"],
    ),
    safety_overrides={"mcp__github__create_issue": ToolCategory.ASK},
)
```

A denial never raises: it returns `{"error": "Permission denied: …", "denied": true}` to the
model through the normal tool-result path and also emits `ToolEventType.PERMISSION_DENIED`
through `on_event` before the call's `TOOL_END`. Callback exceptions fail closed. Denials share
the consecutive cap with errors (5 in a row stops the loop). `safety_overrides` reclassify
individual tools per AgentCore instance — builtins are shared singletons and never mutated.

**Environment axis (where does it run, what does it see?).** Pass `execution_policy=` to the
constructor or per `chat()`/`stream()` call (call-site wins). While ambient: `Bash` and `Exec`
run with the policy env (`allowlist`/`denylist`/inherit) and pinned cwd — `Bash` logs an INFO
line when the pin overrides its `directory` argument; `Read`/`LS`/`Grep`/`Write` enforce
`read_paths`/`write_paths` (resolved against the policy cwd, symlinks followed). `run_skill`
applies a call-site policy verbatim; otherwise an agent-level policy is reused with cwd pinned
to the skill's directory.

The two axes are complementary, not a hard sandbox: `read_paths`/`write_paths` bind the file
tools only — an approved `Bash`/`Exec` call runs as your process user and can touch anything
that user can, under the filtered env and pinned working directory. Use the permission axis
(callback, `safety_overrides`) to decide whether shell access is granted at all; the paths are
a guardrail against accidental access, not an OS-level confinement.

## License

Apache-2.0
