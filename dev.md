# CLIver Development

## Overview

CLIver is a **Personal AI Lab** for AI agent research and experimentation. It provides a Re-Act loop engine (AgentCore), 22 built-in tools, a skills system, MCP integration, and a Gateway daemon — all from a single `~/.cliver/config.yaml`.

## Quick Start

```bash
make init          # venv + dependencies
make test          # run tests (474 passing)
make lint          # ruff check + format
make format        # auto-fix

uv run cliver      # interactive mode
uv run pytest      # run all tests
```

## Architecture

```
src/cliver/
├── llm/agent_core.py     # Re-Act loop: chat(), stream(), generate()
├── agent_factory.py      # create_agent_core(model_config) — auto-wires everything
├── system_prompt.py      # Stateless prompt builder
├── config.py             # AppConfig, ProviderConfig, ModelConfig (Pydantic + YAML)
├── provider/             # OpenAI + Anthropic protocol engines
├── tool.py               # CLIverTool, ToolRegistry, @tool decorator
├── tools/                # 22 builtin tool implementations
├── skill_manager.py      # SKILL.md discovery from 7 sources
├── mcp/                  # MCP client + adapters
├── permissions.py        # PermissionManager (global + project-local)
├── session_manager.py    # SQLite session persistence
├── task_manager.py       # CRUD for scheduled tasks
├── agent_profile.py      # CliverProfile (identity + preferences)
├── key_store.py          # Encrypted secret storage (Fernet)
├── ui_bridge.py          # UIBridge protocol + CLIBridge + TUIBridge
├── messages.py           # CLIverMessage, CLIverMessageChunk, ToolCall
├── cli.py                # Cliver class, Click group, main entry
├── commands/             # CLI subcommands (config, model, session, task, etc.)
├── gateway/              # Daemon, cron scheduler, admin API, Slack adapter
└── skills/               # 4 builtin skills (brainstorm, write-plan, execute-plan, wbs-planner)
```

## Website

```
website/                  # Astro 5 + Starlight
├── src/pages/            # Homepage, blog
├── src/content/docs/     # Documentation (MDX)
├── src/components/       # Custom Header, PageTitle
└── src/styles/           # Deep Tech CSS theme
```

```bash
make docs-serve           # Dev server at localhost:4321
make docs-build           # Static output to website/dist/
```

## Config Directory

```
~/.cliver/
├── config.yaml           # Providers, models, gateway, session
├── cliver-settings.yaml  # Permission mode + rules
├── identity.md           # Agent profile (name, role, preferences)
├── memory.md             # Persistent knowledge
├── skills/               # User-global skills
├── tasks/                # Task definitions
└── audit_logs/           # Token usage logs
```

## Key Design Decisions

- **No langchain.** AgentCore implements its own Re-Act loop.
- **Stateless AgentCore, stateful calls.** One instance per model. Conversation context passed per call.
- **One system prompt.** All system content as a single SystemMessage.
- **Config via YAML.** Secrets resolved: KeyStore → env var → literal.
- **Tools sorted alphabetically.** Stable system prompt for LLM caching.

## Research Gaps

These are the missing pieces for making CLIver a proper AI agent research platform:

### 1. Evaluation Framework
No way to run the same prompt suite across multiple models/configs and compare results. A researcher needs: define a test suite, run it against model A and model B, see which performs better.

### 2. Experiment Reproducibility
No way to capture a full experiment as a portable artifact — model, tools, skills, system prompt, and test inputs in one file that can be shared or re-run later.

### 3. Structured Tracing & Dashboards
`on_event` gives raw events, but there's no built-in visualization for agent traces, tool call latency, or error pattern analysis across runs.

### 4. Custom Tool Registration Path
The `@tool` decorator exists in code, but there's no documented workflow for researchers to add their own tools without modifying the source.

### 5. Multi-Agent Orchestration
No way to run multiple agents in parallel or have them collaborate — essential for experimenting with agent architectures beyond single-agent loops.

### 6. Sandbox Execution
No isolated environment (container/VM) for safely running untested agent configurations or third-party skills.

### 7. Cost-Performance Correlation
Token tracking works, but there's no way to answer: "did spending 2x more tokens actually improve task completion?"

### 8. Agent State Checkpointing
Can't snapshot an agent mid-experiment and restore it later — important for debugging and A-B testing from the same starting state.
