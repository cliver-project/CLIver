# Changelog

All notable changes to CLIver will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
