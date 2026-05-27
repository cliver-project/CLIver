---
title: Roadmap
description: Development status and future plans for CLIver
---

# Roadmap

## Current Release: v0.1.4 (Alpha)

### Completed
- 17 built-in tools (file I/O, shell, web, planning, memory, identity, skills)
- Multi-provider LLM support (OpenAI-compatible, Ollama, vLLM)
- MCP server integration
- Skills system with LLM-driven activation
- Layered permission system (default, auto-edit, yolo)
- Memory and identity profiles with multi-agent isolation
- Session management with LLM-based compression
- Cost tracking per provider and model
- Gateway daemon with Telegram, Discord, Slack, Feishu adapters
- Docker support (GHCR)
- Embeddable AgentCore Python API

### In Progress
- Documentation improvements
- Media generation support (image, audio)

### Planned
- **Deep Search** — `cliver deep-search "question"` multi-step research
- **Multi-Model Orchestration** — route different capabilities to different models
- **Plugin System** — installable capability packages

## Contributing

See [CONTRIBUTING.md](https://github.com/cliver-project/CLIver/blob/main/CONTRIBUTING.md) for how to get involved.

Have an idea? Open a [feature request](https://github.com/cliver-project/CLIver/issues/new?template=feature_request.md) or start a [discussion](https://github.com/cliver-project/CLIver/discussions).
