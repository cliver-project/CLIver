# Changelog

All notable changes to CLIver will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **cliver-core: `ModelBinding.for_current_loop()`** — returns a sibling binding that shares the
  model definition, params, and a caller-supplied provider but resets the event-loop claim, so an
  application can keep one binding spec and use it from multiple loops/threads. A binding that
  builds its provider builds a fresh one on the clone's first loop; a caller-supplied provider
  that owns loop-bound resources is refused.
- **cliver-core: `LoopAffinityError`** — the typed `RuntimeError` subclass raised when a binding
  with loop-bound resources is used from a second event loop; existing `except RuntimeError`
  handlers keep working.
- **cliver-core: `Provider.loop_agnostic`** — a provider class that owns no loop-bound resources
  declares `loop_agnostic = True`. The in-process local embed/rerank providers do.
- **cliver-core: `AgentCore.open(bindings, **agent_kwargs)` and async context-manager support** —
  `async with AgentCore.open(bindings) as agent:` awaits `aclose()` on exit, on normal and
  exceptional paths alike. `AgentCore(bindings)` itself is also an async context manager. Close
  remains idempotent and single-flight, and still never closes a caller-supplied provider or the
  MCP client.
- **cliver-core: `ModelBinding.build_provider()`** — the documented name for the idempotent
  "build once for this loop" call. `get_provider()` remains as a compatibility alias for 0.3.x
  callers.
- **cliver-core: `ModelBindings.preflight()` / `ModelBinding.preflight()`** — builds every
  configured provider (primary and fallbacks) on the calling loop and resolves each
  `credential_provider` once, so a missing endpoint or credential raises at startup instead of
  on the first call. No network request; idempotent; a failed build leaves the binding
  unchanged. Lazy construction remains the default.
- **cliver-core: `cliver_core.integrations.lifespan`** — `agent_lifespan(bindings, **agent_kwargs)`
  returns a framework-agnostic ASGI lifespan callable that builds the agent, preflights the
  bindings, publishes the agent on `app.state.agent`, and closes it on every exit path. The
  lifespan yields nothing (the ASGI lifespan yield is reserved for a state mapping), so read the
  agent through `get_agent(request)` from an application-side `Depends` wrapper. No FastAPI or
  Starlette import.

### Fixed

- **cliver-core: `aclose()` keeps the event-loop claim of a supplied loop-bound provider** — the
  bindings' claim is now released only when a binding can rebuild its provider (the one it built
  was just closed) or holds only `loop_agnostic` providers. A caller-supplied provider stays
  open on its original loop, so reusing that binding from a second loop now raises
  `LoopAffinityError` instead of quietly handing out a client bound to a dead loop.
- **cliver-core: loop-agnostic providers no longer trip the binding's event-loop guard** — a
  binding wrapping a caller-supplied `LocalEmbeddingProvider` / `LocalRerankProvider` (or any
  `loop_agnostic` provider) no longer claims the loop it is first used on, so it can stay cached
  while calls run under consecutive `asyncio.run()` calls in one process. Bindings that build a
  remote SDK provider keep the guard unchanged. See
  [Concurrency & Lifecycle](https://cliver.dev/core-concepts/concurrency/).

## [0.3.0] - 2026-11-02

The first stable release of the 0.3 series. The public `cliver-core` API is frozen; only bug
fixes land in `0.3.x` patch releases. The `cliver` CLI remains experimental and is not
published to PyPI.

### Added

- **cliver: interactive REPL** — running `cliver` with no prompt on a terminal now starts a
  streaming, multi-turn session instead of erroring. Each turn keeps the conversation for
  follow-ups; `/help`, `/clear`, `/exit` (or `/quit`) and Ctrl-D manage the session. Tool
  events and the SAFE/ASK permission gate behave exactly as in one-shot mode. Piped stdin and
  explicit prompts still run one-shot. Bare `exit` / `quit` also leave the session.
- **cliver: clean failure reporting** — configuration errors print a one-line
  `cliver: error: ...` (exit 2) instead of a traceback; provider failures print the error plus
  the endpoint and protocol used (exit 1), and Ctrl-C exits 130. In the REPL, provider errors
  include the same endpoint line and keep the session alive.
- **cliver-core: declarative context compaction** — `context_compactor` may now return a
  `CompactionPlan` instead of a complete history. Core validates the `keep_from` boundary
  (inside the history, pointing at a `user` message), prepends the summary as
  `[Compacted earlier context]`, preserves tool call/result pairs, and records the plan's usage
  into `usage_total`. New `CompactionContext.split_at_user_turn()` / `.render()` and
  `CompactionSplit.plan()` keep the boundary bound to the text the summarizer saw. Returning a
  full `list[CLIverMessage]` remains supported as the escape hatch.
- **cliver-core: `LlmCompactor`** — an opt-in, threshold-triggered summarization policy built on
  plans. It is never installed automatically; applications must pass it to
  `AgentCore(context_compactor=...)`.

### Upgrade Notes from 0.3.0rc1

- Pin exactly: `pip install cliver-core==0.3.0`.
- No public API changes since `0.3.0rc1`; the additions are backward-compatible.

## [0.3.0rc1] - 2026-11-01

The 0.3 release candidate. The public `cliver-core` API is frozen: only bug fixes land before
`0.3.0`. The `cliver` CLI remains experimental and is not published to PyPI.

### Added

- **cliver-core: metered inference events** — every `inference.completed` event now includes the
  call's usage under `data["usage"]` (`UsageInfo` dump, or `None` when the vendor reported
  none), for both `chat()` and `stream()` on the OpenAI and Anthropic protocols. Per-call
  metering works from events alone, across multi-iteration Re-Act runs.
- **cliver-core: `CLIverResponse.usage_total`** — the sum of every model call in the run,
  intermediate tool-requesting iterations included. `response.usage` keeps its existing
  meaning: the final model call's usage.
- **cliver-core: binding-level inference events** — `ModelBinding.on_event` /
  `create_provider(on_event=...)` subscribes a provider to inference events at construction, so
  events are reachable without supplying a pre-built provider. Combining `provider=` with a
  binding-level `on_event` raises a configuration error (the supplied instance keeps its own).
- **cliver-core: `ProviderError`** — provider boundaries raise one error type with the vendor's
  exception preserved as `__cause__` and a best-effort `retryable` flag (rate limits, timeouts,
  connection failures, and 5xx responses are retryable). `inference.error` reports it in
  `data["retryable"]`.
- **cliver-core: DashScope, Qwen, and Qwen-Image providers** — the `dashscope` brand runs native
  asynchronous image/video tasks on `/api/v1` (create, poll, deadline handling, `oss://`
  reference uploads), `qwen` provides chat on the compatible-mode endpoint, and `qwen-image`
  implements the synchronous reference-aware image wire with direct data-URI inputs.
  `presets.qwen` and `presets.qwen_image` are live-verified; the DashScope media presets are
  implemented and covered by the offline test suite.
- **cliver-core: same-purpose failover** — every purpose accepts an ordered list of bindings:
  the first is primary and the rest are fallbacks used only for retryable provider failures.
  Failover never crosses purposes; chat is sticky after a switch; streams switch only before
  the first chunk; media returns the last friendly failure and embed/rerank raise the last
  error.
- **cliver-core: parallel tool execution and per-tool deadlines** — opt-in
  `parallel_tool_calls=True` runs tool bodies concurrently while permission checks stay
  sequential and results preserve call order. `tool_timeout` is available on the constructor
  and per call (omit to inherit, `None` to disable); async tools and MCP calls run on the
  agent's loop.
- **cliver-core: explicit execution scope and per-run tool state** — the advertised tool set is
  the execution set; tools outside it are rejected before permission checks, and
  `mcp_servers=[]` disables MCP for a call. Per-run `ToolRunState` gives Todo tools run-scoped
  state instead of agent-scoped state.
- **cliver-core: System One decision models** — `ModelBinding.system_one(...)` mounts the
  Jev-compatible decision wire; `Choice`, `Score`, and `Noul` return probabilities, labels, and
  scores rather than generated text. Hosted Bailian and TypeSafe presets plus the local `laya`
  backend are supported, and `SystemOneSelector` uses the decision model to narrow tools and
  skills as a fail-open candidate selector.
- **cliver-core: FastAPI harness primitives** — `run_id` correlation across inference, tool,
  pipeline, and terminal stream events; `run_timeout` with `RunTimeoutError`; explicit
  credential rotation through `arefresh()` / `arefresh_credentials()`; `chat_structured()` with
  pydantic or JSON-Schema dict output, validation, retries, and usage accounting; a streamable
  HTTP MCP session pool (`session_pool_size`); and `client_timeout` / `client_max_retries` SDK
  client options. The published FastAPI guide covers the wiring.
- **cliver-core: caller-supplied context compactor** — the `context_compactor` hook receives a
  `CompactionContext` once per Re-Act iteration, before the model call, and returns a validated
  replacement history. Compaction runs inside the run deadline and its model usage is counted;
  `None` disables it.
- **cliver-core: Bailian workspace hosts** — `qwen`, `qwen_image`, `dashscope_image`,
  `dashscope_video`, and `bailian_system_one` accept `workspace_id=` / `region=` and compose
  the workspace-dedicated `maas.aliyuncs.com` host; an explicit `base_url=` still wins, and no
  workspace id keeps the shared DashScope default.
- **cliver-core: committed live verification suite** — `tests/live/` runs 23 checks against real
  providers under `CLIVER_LIVE_VERIFICATION=1` / `make live-verify`, with missing provider keys
  warning and skipping rather than failing.

### Changed

- **cliver-core: sync event handlers** — `EventHandler` accepts sync and async callables; a
  sync handler runs inline. Handler exceptions still propagate unchanged (a raise on
  `inference.started` happens before the transport call).
- **cliver-core: `AgentCore.on_event` is tool-only** — instance and per-call handlers receive
  tool events; inference events come from the binding/provider `on_event`.
- **cliver-core: lifecycle contract** — `AgentCore.aclose()` drains media jobs it started and
  closes providers its bindings built, but never owns the MCP client or a caller-supplied
  provider. Bindings are loop-affine: create and use them on the same event loop.
- **cliver-core: `api_key` is excluded from binding reprs** — accidental logging no longer
  prints credentials.
- **cliver: hosted System One model is optional** — `CLIVER_SYSTEM_ONE_MODEL` can be omitted for
  the `bailian` and `typesafe` presets, which pin a default; it remains required for `laya` and
  custom endpoints.

### Fixed

- **cliver-core: streamed usage is staged before the terminal event** — the streaming terminal
  chunk's usage is retained locally, so `inference.completed` for a stream carries the same
  usage as the chunk it summarizes.

### Breaking Changes

- **Execution scope** — `AgentCore` advertises exactly the tools it will execute. A tool that is
  not in the call's execution set is rejected before permission checks; there is no registry
  fallback. Pass the intended tools explicitly, and use `mcp_servers=[]` to disable MCP.
- **Loop affinity** — bindings, agents, and MCP clients belong to the event loop that created
  them. Using them from another loop raises `RuntimeError`; thread safety is not provided.
- **`bailian_system_one()` requires an endpoint** — pass `workspace_id=` or `base_url=`;
  calling it with neither now raises `ValueError`.
- **Provider exceptions are wrapped** — catch `ProviderError` and inspect `.retryable` rather
  than relying on raw vendor exception types. It subclasses `RuntimeError` for compatibility
  with older embed/rerank callers.

### Upgrade Notes from 0.3.0a5

- Pin exactly: `pip install cliver-core==0.3.0rc1` (a bare install will not select a release
  candidate).
- If a call relied on tools being advertised from the registry without being passed, pass the
  execution set explicitly.
- Create bindings and agents inside the event loop that serves the request; do not share them
  across threads or loops.
- Pass `workspace_id=` or `base_url=` to `bailian_system_one()`.
- Replace raw provider exception handling with `ProviderError`; use `.retryable` to decide
  whether to retry or fail over.

## [0.3.0a5] - 2026-10-25

### Added
- **cliver-core: deterministic skill pipelines** — `SkillPipeline` runs an ordered list of
  `PipelineStep`s sequentially, each in its own fresh context (a skill via `run_skill()`, a raw
  prompt via `chat()`). The caller defines the steps; no LLM decides the sequence. Each step's
  input is composed from the original input and prior results (`input_builder`, sync or async),
  results expose `content`, optional `metadata`, `usage`, and `error`, and failures stop the
  pipeline by default or continue per step (`FailurePolicy`). A synthesis step is an ordinary
  final step that reads the earlier outputs. Per-step `runner`s may use different `AgentCore`
  instances, models, tools, or skill sets; `AgentCore` stays stateless. The pipeline emits
  `PipelineEvent`s (`pipeline.start`, `pipeline.step_start`, `pipeline.step_end`,
  `pipeline.step_error`, `pipeline.end`) and forwards each step's `ToolEvent`s; cancellation
  emits `pipeline.end` with `{"cancelled": true}` and re-raises. Exported from `cliver_core`.
- **cliver-core: per-call event handlers** — `chat()`, `stream()`, and `run_skill()` accept an
  `on_event` handler that runs after the instance handler and receives the same tool events,
  so a caller (such as a pipeline step) can observe a run without mutating the agent.

## [0.3.0a4] - 2026-10-25

### Added
- **cliver: one-shot CLI application** — `cliver` now ships an executable one-shot command on
  top of the core. It wires the eight skill sources and nine builtin tools, gates mutating
  tools through a SAFE/ASK permission callback (`--yes` approves, an interactive terminal
  prompts, non-interactive runs deny with a hint), reports tool activity on stderr, and reads
  the endpoint from `--model` / `CLIVER_MODEL`, `--provider` / `CLIVER_PROVIDER` (or
  `--base-url` / `CLIVER_BASE_URL`), `--protocol`, and `--api-key` / `CLIVER_API_KEY`.
  `--skill` runs a skill explicitly, `--cwd` selects the project root, piped stdin works as
  the prompt, and `{config_dir}/.env` is loaded without overriding the process environment.
  Both packages now share one version, guarded by `test_versions_aligned.py`.

### Changed
- **cliver-core: toolset gating removed** — `ToolRegistry` now advertises exactly the tools
  that were registered. The stale `TOOLSETS` table (it still named ten tools that no longer
  exist, plus a Firecrawl environment check), `ToolRegistry.configure()`, and the
  `enabled_toolsets` parameter are gone. Tool selection is the application's decision: pass
  `builtin_tools=discover_builtin_tools()` or a narrowed list.

### Fixed
- **cliver: brand API key fallback** — the CLI reads a known provider's conventional key
  variable (`DEEPSEEK_API_KEY`, `MINIMAX_API_KEY`, `GLM_API_KEY`, …) when `CLIVER_API_KEY` is
  unset, so a key in `~/.cliver/.env` works without extra wiring.

## [0.3.0a3] - 2026-10-25

### Added
- **cliver-core: tool permission + execution policy** — two opt-in axes around tool execution.
  Permission: every tool carries a category (SAFE/ASK — read-only builtins safe, mutating ones
  ask, MCP `readOnlyHint` maps to safe as a hint only); an async `permission_callback` gates
  ASK tools before they run, returning `{"error": "Permission denied: …", "denied": true}` to
  the model through the normal tool-result path. Callback exceptions fail closed; denials
  share the consecutive cap with errors; `safety_overrides` reclassify tools per AgentCore
  instance without mutating the shared singletons. Environment: `ExecutionPolicy`
  (`EnvSpec` allowlist/denylist/inherit, pinned cwd, `read_paths`/`write_paths`) is ambient
  per `chat()`/`stream()` call (call-site over constructor default) — `Bash`/`Exec` honor env
  and cwd (the pin overriding Bash's `directory` is logged at INFO), `Read`/`LS`/`Grep`/
  `Write` enforce scopes with symlinks followed. `run_skill` applies a call-site policy
  verbatim, else reuses the agent policy with cwd pinned to the skill directory.
  Exported from `cliver_core` (`Decision`, `ExecutionPolicy`, `EnvSpec`, `SAFE_ENV_VARS`, …).
- **cliver-core: Agent Skills** — `cliver_core.skill.SkillManager` loads SKILL.md skills
  (agentskills.io spec) from explicit application-supplied sources with a dependency-free
  frontmatter parser (flat scalars, quoted strings, `>`/`|` blocks, nested maps of any depth —
  cross-agent metadata keys like `x-workbuddy:` parse into `skill.frontmatter`) and spec
  validation as load-time warnings. `AgentCore.run_skill(skill_or_name, message, ...)` appends
  the skill body to the single system message and runs the Re-Act loop with the skill's
  `allowed-tools` enforced (failing fast when none are available); every `chat()` parameter
  passes through. Bundled files are listed (`skill.files`) and readable on demand
  (`skill.read_file`, path escapes rejected). The activation header carries the skill's
  location so the model can run bundled scripts through an allowed execution tool. Script
  execution sandboxes remain future work.
- **cliver-core: model-driven skill selection** — with a `SkillManager` configured, `chat()` /
  `stream()` expose skills through three levels: a system-prompt summary of a curated subset
  (name + shortened description, `skill_summary_limit` default 10), `FindSkills(query, limit)`
  to search the full catalogue (an empty query lists every name), and `ActivateSkill(name)` to
  load a skill's body as a tool result — the same framing as `run_skill`, location included.
  Activation is per run: one skill at a time, and a declared `allowed-tools` narrows the
  advertised tools for the remaining iterations (an allowlist matching no available tool fails
  the activation and leaves the run unchanged). Core stays embedded: the app owns sources via
  `SkillManager` and may supply `skill_selector(skills, user_input)` to rank the summary
  (popularity today, encoder / system-1 relevance later) and `skill_search(query, skills)` to
  replace the default lexical search; both seams accept sync or async callables and default to
  dependency-free behavior. `run_skill()` disables dispatch for its run (`skill_dispatch=False`);
  a regular tool named `FindSkills` or `ActivateSkill` raises while dispatch is on (reserved
  names).
- **cliver-core: query-scoped candidate guardrail** — skills no longer compete with tools for
  the same intent. The skill summary now states the precedence rule (use a tool when one can
  do the task; activate a skill only when no tool can, or when the task needs the skill's
  specific procedure), and `ActivateSkill`'s description says the same. Two optional,
  app-owned seams narrow what a call advertises, both accepting sync or async callables:
  `candidate_selector(tools, skills, user_input)` sees both candidate pools and returns both
  — the cross-category guardrail, e.g. suppressing a skill a tool already covers — and
  `tool_selector(tools, user_input)` narrows tools alone. They compose with the per-call
  `tool_filter` and the existing `skill_selector`; `run_skill()` applies the same pipeline to
  its allowlist check. Core defaults stay permissive and dependency-free — the filtering
  policy (lexical, encoder-only, system-1) lives in the application.
- **cliver-core: skill-scoped file reads** — activating a skill that bundles files
  (`references/`, `scripts/`, `assets/`, ...) now also advertises `ReadSkillFile(path=...)`,
  confined to the active skill's directory: absolute paths, `../` escapes, and symlink
  escapes are refused, content must be UTF-8 text, and files are capped at 256 KiB. The
  activation result lists the bundled files and says to use `ReadSkillFile` rather than the
  workspace `Read` tool; the tool is advertised only when the skill actually has files and
  survives `allowed-tools` narrowing, so a skill activated via `ActivateSkill` can allow only
  `ReadSkillFile` and still read its own references. `run_skill()` deliberately adds none of
  the model-driven skill tools — its cwd is already pinned to the skill directory. The tool
  grants no general filesystem access — workspace read scopes are untouched.
  `Skill.read_file(relative_path, max_bytes=...)` gained the optional size cap.
- **cliver: eight-source skill discovery** — the app package now assembles CLIver's skill
  sources — the builtin skills shipped with `cliver`, `{config_dir}/skills/`,
  `~/.agents/skills/`, and project-local `.cliver/`, `.agent/`, `.claude/`, `.gemini/`, and
  `.qwen/` — into `cliver_core.SkillManager`, later sources overriding earlier ones by name.
  The pre-split builtin skills were restored; source labels stay app-owned, and cliver-core
  remains discovery-free.

### Changed
- **cliver-core: builtin tool discovery** — `cliver_core.tools.discover_builtin_tools()` now
  scans the package (`pkgutil`) and collects every module-level `CLIverTool` from public
  modules (underscore-prefixed modules are skipped; duplicate tool names raise). Adding a tool
  is one new module with `@tool`, no registry edit. The nine tool objects are no longer
  re-exported from `cliver_core.tools` — import them from their modules
  (e.g. `cliver_core.tools.read_file`) or go through discovery.
- **cliver-core: MCP is a base dependency** — the MCP client SDK moved from the optional
  `mcp` extra into the core dependencies (`mcp>=2.3.0,<3.0`). MCP support is first-class:
  a plain `pip install cliver-core` now includes the client, and the adapter tests always
  run instead of skipping when the SDK is absent. The `mcp` and `all` extras are removed
  (`all` only ever meant "with MCP"); the local-model extras (`local-st`, `local-flag`,
  `local`) are unchanged.
- **cliver-core: tool registry probe cleanup** — the registry no longer probes for playwright.
  cliver-core ships no Browser tool and browser availability belongs to the package that ships
  it; the `browser` toolset gate is unchanged.

### Removed
- **cliver-core: dead media module** — `cliver_core/media_handler.py` (never exported, imported,
  or documented; the media pipeline returns `CLIverResponse` and `MediaJob` directly) was
  removed.

### Fixed
- **cliver-core: empty `allowed-tools` in `run_skill()`** — `allowed-tools: []` now behaves
  like an absent list (no narrowing), matching model-driven activation; previously it installed
  a filter that advertised zero tools.
- **cliver-core: permission-denied events** — a denied tool call now emits
  `ToolEventType.PERMISSION_DENIED` before its `TOOL_END`, so `on_event` monitors observe
  denials directly.
- **cliver-core: builtin tool descriptions** — WebFetch no longer references a nonexistent
  `ReadFile` tool (now `Read(file_path=...)`, including in its saved-file summary); Bash
  documents its millisecond `timeout` and `directory`; Exec no longer claims PyYAML is
  available and documents its seconds-based timeout; Bash and Exec now cross-reference each
  other (Python code → Exec, shell commands → Bash); TodoWrite documents the todo item shape;
  Read documents `offset`/`limit` and sensitive-file refusal; Grep/LS wording now matches
  their actual behavior.

## [0.3.0a2] - 2026-10-19

### Added
- **cliver-core: in-process local models** — `cliver_core.local` provides `LocalEmbeddingProvider`
  and `LocalRerankProvider` for sentence-transformers and FlagEmbedding models. A provider is
  long-lived, loads lazily and thread-safely, serializes inference, and runs blocking work off the
  event loop. `local_embed` / `local_rerank` build bindings around a caller-supplied provider;
  `ModelBinding.aclose()` never closes it.
- Explicit `model` + `backend` parameters; no model-name routing, instruction templates, or
  query/document modes in the library. Generic options only: `dim`, `normalize`, `batch_size`, and
  `top_k`.
- `provider.info()` reports the resolved model directory, revision, device, dtype, dimension, and
  load time; `LocalModelError` subclasses classify dependency, load, device, configuration,
  dimension, inference, and closed-provider failures.
- Optional extras: `local-st`, `local-flag`, and `local` (both). The backends pull torch and stay
  opt-in; `all` remains MCP-only.
- Gated live tests for `BAAI/bge-base-zh-v1.5` and `BAAI/bge-reranker-v2-m3`
  (`CLIVER_LOCAL_MODEL_TESTS=1`, model env-overridable).

### Changed
- sentence-transformers 6 renamed `get_sentence_embedding_dimension` to
  `get_embedding_dimension`; the backend prefers the current method and falls back for older
  releases.

## [0.0.3] - 2026-03-24

### Added
- **Skills: Agent Skills spec compliance** — SKILL.md now follows the [agentskills.io](https://agentskills.io/specification) specification with validation for name, description, license, compatibility, metadata, and allowed-tools fields
- **Skills: Multi-source discovery** — Skills are discovered from 7 locations: `{config_dir}/skills/`, `~/.agents/skills/`, `.cliver/skills/`, `.agent/skills/`, `.claude/skills/`, `.gemini/skills/`, `.qwen/skills/`
- **Skills: Cross-agent compatibility** — Tolerant loading of skills from Claude Code, Gemini, and Qwen Code (accepts non-standard names, logs warnings)
- **Agent management** — `/agent` command with create, switch, list, rename, delete subcommands
- **Animated thinking indicator** — Shows model name and thinking phrases during LLM inference
- **Task cancellation** — Ability to cancel running tasks

### Changed
- **Memory curation** — Memory is now guided toward topic-based organization instead of chronological append-only logging; LLM is instructed to periodically consolidate with rewrite mode
- **Skill list output** — `skill('list')` now shows skill count, source labels, truncated descriptions, and activation hint
- **CI pipeline** — Added formatting check (`ruff format --check`) to CI workflow
- **Build tooling** — `make format` now runs `ruff check --fix` before `ruff format`

### Fixed
- Default model sync to AgentCore when changed via `/model default`
- Centralized output and TUI layout improvements

## [0.0.2] - 2025-10-27

### Added
- **Permission system** — Layered permission model with persistent rules, session grants, and task-scoped overrides; three modes: default, auto-edit, yolo
- **Conversation sessions** — Session management with save/load, conversation history, and LLM-based compression
- **Token tracking** — Per-model token usage tracking with `/cost` command and audit logging
- **Workflow engine** — Multi-step workflow execution with function, LLM, human, and nested workflow steps; pause/resume/cache support
- **Skill system** — LLM-driven skill activation with SKILL.md format and progressive disclosure
- **Memory & Identity** — Persistent agent memory and identity profiles across sessions
- **Agent profiles** — Instance-scoped resource management for multi-agent isolation
- **Planner** — Built-in planning with todo_read/todo_write and complexity-based approach selection
- **Task scheduling** — Task definitions with workflow binding and cron scheduling
- **16 builtin tools** — read_file, write_file, list_directory, grep_search, run_shell_command, todo_read, todo_write, memory_read, memory_write, identity_update, ask_user_question, skill, web_search, web_fetch, docker_run, setup_docker
- **Tool registry** — Keyword-based filtering to reduce token usage
- **Secret resolver** — Jinja2 templates with keyring and environment variable support
- **Status bar** — Shows cwd, permission mode, model, and token usage
- **Streaming** — Enabled by default for chat command
- **Multi-line input** — Ctrl+G for editor support
- **DeepSeek engine** — Subclass with reasoning_content preservation

### Fixed
- Consecutive error detection in Re-Act loop (MAX_CONSECUTIVE_ERRORS=3)
- Tool errors sent back to LLM instead of stopping the agent loop
- Keyring secret caching to avoid repeated password prompts

## [0.0.1] - 2025-10-13

### Added
- Initial release
- CLI framework with Click
- LLM integration with OpenAI, Ollama, and vLLM providers
- MCP server integration via langchain-mcp-adapters
- YAML-based configuration
- Basic chat command with Re-Act pattern
