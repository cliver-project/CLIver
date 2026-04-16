# CLIver

**CLIver** is a general-purpose AI agent for your command line. It is not tied to any specific domain — with customizable system prompts, skills, workflows, and MCP integrations, you can adapt it to any task: coding, DevOps, research, writing, data analysis, or anything else you need.

CLIver is also built to be **safe and controlled**. A layered permission system governs every tool execution, and a structured workflow engine keeps complex tasks on track — so you get the power of autonomous AI without scattered, unpredictable behavior.

## Quick Start

```bash
# Install via pip
pip install cliver

# Or run with Docker
docker run --rm -it --user $UID:0 -v ~/.cliver:/home/cliver/.cliver \
  -e OPENAI_API_KEY ghcr.io/cliver-project/cliver

# Start chatting
cliver chat "What time is it in Beijing and London?"

# Interactive mode
cliver chat
```

## Key Features

- **Multi-provider LLM support** — OpenAI-compatible (Qwen, DeepSeek, etc.), Ollama, vLLM
- **MCP integration** — Connect to any MCP server for extended tool capabilities
- **Skills** — LLM-driven skill activation following the [Agent Skills](https://agentskills.io) specification
- **Memory & Identity** — Persistent knowledge and agent profiles across sessions
- **Permissions** — Layered tool permission system (default, auto-edit, yolo modes)
- **Workflows** — Multi-step workflow engine with pause/resume support
- **Embeddable** — `AgentCore` API can be used as a Python library

## Documentation

Full documentation is available at the [docs site](https://cliver-project.github.io/CLIver/):

- [Installation](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Chat Usage](docs/chat.md)
- [Skills](docs/skills.md)
- [Memory & Identity](docs/memory-identity.md)
- [Permissions](docs/permissions.md)
- [Workflows](docs/workflow.md)
- [Extensibility](docs/extensibility.md)

## Development

```bash
make init      # Set up dev environment
make test      # Run tests
make lint      # Lint + format check
make format    # Auto-fix lint and formatting
```

## License

See [LICENSE](LICENSE) for details.
