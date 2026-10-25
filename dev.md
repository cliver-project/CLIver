# CLIver Development

## Overview

CLIver is a **Personal AI Lab** for AI agent research and experimentation. The repository holds
two packages:

- **`cliver-core`** — the embeddable agent engine: Re-Act loop, provider protocol engines,
  per-purpose model bindings, builtin tools, skills, MCP, and opt-in policy. Zero CLI
  dependencies.
- **`cliver`** — the CLI application built on `cliver-core`.

## Quick Start

```bash
make init          # venv + workspace dependencies
make test          # run both packages' tests
make lint          # ruff check + format check
make format        # auto-fix lint/format issues

uv run pytest      # run all tests directly
```

## Architecture

```
packages/cliver-core/src/cliver_core/
├── agent.py            # AgentCore: chat(), stream(), run_skill()
├── bindings.py         # ModelBindings / ModelBinding
├── presets.py          # live-verified provider bindings
├── provider/           # OpenAI + Anthropic protocol engines, media pipeline
├── tool.py             # CLIverTool, ToolRegistry, @tool decorator
├── tools/              # nine builtin tools (discovered by scanning)
├── skill.py            # SKILL.md parsing + SkillManager
├── skill_tools.py      # FindSkills / ActivateSkill / ReadSkillFile
├── mcp/                # MCP client + adapters
├── policy.py           # Decision, ExecutionPolicy, scopes
├── capabilities/       # image / video / tts / asr / embed / rerank APIs
├── local/              # optional in-process embed/rerank backends
└── tests/

packages/cliver/src/cliver/
├── paths.py            # config-directory resolution
└── skills/             # eight-source discovery + four builtin skills
```

## Website

```
website/                  # Astro 5 + Starlight
├── src/pages/            # Homepage
├── src/content/docs/     # Documentation (MDX)
├── src/components/       # Custom Header, PageTitle
└── src/styles/           # Deep Tech CSS theme
```

```bash
cd website && npm install
npm run dev               # Dev server at localhost:4321
npm run build             # Static output to website/dist/
```

## Config Directory

The CLI application resolves its config directory from `CLIVER_CONF_DIR`, `%APPDATA%/cliver`
on Windows, or `~/.cliver` otherwise. Skills come from eight sources: builtin,
`{config_dir}/skills`, `~/.agents/skills`, and project-local `.cliver`, `.agent`, `.claude`,
`.gemini`, `.qwen`.

## Key Design Decisions

- **No langchain.** AgentCore implements its own Re-Act loop.
- **Stateless AgentCore, stateful calls.** One instance per model; conversation context is
  passed per call.
- **One system prompt.** All system content as a single SystemMessage.
- **Core stays embeddable.** App-specific behavior and knowledge live outside `cliver-core`.
- **Tools sorted alphabetically.** Stable system prompt for LLM caching.
