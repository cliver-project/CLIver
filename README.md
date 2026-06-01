<div align="center">
  <img src="icon.png" alt="CLIver" width="120"/>
  <h1>CLIver</h1>
  <p><strong>Personal AI Lab</strong> — Experimenting with AI agents in your terminal, as a Python library, or as a gateway service.</p>
</div>

---

CLIver is an **AI Agent research and experimentation project**. It provides the building blocks to explore agent architectures — Re-Act loops, tool calling, skill activation, memory systems, and multi-agent orchestration — through a terminal CLI and an admin web UI.

**A lab, not a product.** CLIver is designed for tinkering. Swap models mid-conversation. Test how different system prompts change behavior. Add custom skills and tools to see what sticks.

**Full-stack agent experimentation.** Run interactively in the terminal, embed `AgentCore` as a Python library in your own experiments, or deploy as a gateway service with cron scheduling and messaging platform adapters.

## Quick Start

```bash
pip install cliver

# Configure your LLM provider
cliver provider add --name deepseek --type openai \
  --api-url https://api.deepseek.com --api-key "DEEPSEEK_API_KEY"

# Configure a model
cliver model add -n deepseek-v4-flash --provider deepseek

# Start experimenting
cliver "What can you help me with?"
```

## Built-in Capabilities

- **Model-agnostic** — OpenAI + Anthropic protocols, any provider, switch on the fly
- **Built-in tools** — file I/O, shell, web search, browser, memory, todo tracking, image generation
- **Skills system** — LLM-activated domain expertise following the Agent Skills spec
- **Permission control** — default, auto-edit, and YOLO modes with per-tool overrides
- **Persistent memory** — remembers preferences and context across sessions
- **Gateway mode** — daemon with cron scheduling, admin web UI, and messaging platform adapters
- **Embeddable API** — `AgentCore.chat()` and `.stream()` for your own Python experiments

## Documentation

Full documentation at **[cliver-project.github.io/CLIver](https://cliver-project.github.io/CLIver/)**

## Development

```bash
git clone https://github.com/cliver-project/CLIver.git
cd CLIver

make init     # Set up dev environment
make test     # Run tests
make lint     # Lint and format check
make format   # Auto-fix lint/format issues

make docs-serve  # Start docs website (hot reload)
make docs-build  # Build docs to static files
```

## License

[Apache 2.0](LICENSE)
