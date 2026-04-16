# Changelog

All notable changes to CLIver will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
