# CLIver

An AI agent that makes your CLI clever and more.

CLIver integrates with MCP servers and various LLM providers to bring intelligent capabilities to your terminal — chat, plan, search, code, and automate with natural language.

## Quick Start

```bash
# Install
pip install cliver

# Or install from source
uv sync --all-extras --dev --locked

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
- **Embeddable** — `TaskExecutor` API can be used as a Python library

## Documentation

Full documentation is available at the [docs site](docs/index.md):

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
