<div align="center">
  <img src="https://github.com/cliver-project/CLIver/blob/main/website/public/icon.png?raw=true" alt="CLIver" width="120"/>
  <h1>CLIver</h1>
  <p><strong>Personal AI Lab</strong> — experimenting with AI agents from the terminal or as an embeddable Python library.</p>
</div>

---

CLIver is an **AI agent research and experimentation project** built from two packages:

- **`cliver-core`** — the embeddable engine: a Re-Act loop with tool calling, provider engines
  for the OpenAI and Anthropic protocols, per-purpose model bindings (chat, image, video, TTS,
  ASR, embed, rerank), nine built-in tools, Agent Skills, MCP, and opt-in permission and
  execution-policy axes. No CLI, database, or config files — Python objects in, messages out.
- **`cliver`** — the CLI application built on `cliver-core`. The application package currently
  provides CLIver's config-path and skill-discovery conventions; the executable and TUI are
  still being built, so `cliver-core` is the usable surface today.

**A lab, not a product.** CLIver is designed for tinkering. Swap models mid-conversation. Test
how different system prompts change behavior. Add custom skills and tools to see what sticks.

## Quick Start

Install the core library (the project is currently on pre-releases — pin the exact version):

```bash
pip install cliver-core==0.3.0a3
```

```python
import asyncio

from cliver_core import AgentCore, ModelBindings, presets


async def main():
    agent = AgentCore(ModelBindings(llm=presets.deepseek(api_key="...")))
    response = await agent.chat("Summarize the Re-Act pattern in one sentence.")
    print(response.message.content)
    await agent.aclose()


asyncio.run(main())
```

See the [cliver-core README](packages/cliver-core/README.md) for tools, skills, MCP, media
capabilities, and the full API.

## Built-in Capabilities

- **Model-agnostic** — OpenAI and Anthropic protocols, live-verified presets, any compatible
  endpoint
- **Built-in tools** — file I/O, shell, code execution, grep, directory listing, todo tracking,
  and web fetch
- **Skills system** — Agent Skills loading plus model-driven `FindSkills` / `ActivateSkill` /
  `ReadSkillFile`
- **MCP support** — stdio and streamable HTTP clients with CLIver-style discovery
- **Permission and execution policy** — opt-in SAFE/ASK gates and env/cwd/path scopes
- **Media capabilities** — image, video, text-to-speech, speech-to-text, embedding, and rerank
- **Embeddable API** — `AgentCore.chat()`, `.stream()`, and `.run_skill()` with no CLI
  dependencies

## Repository Layout

```
packages/cliver-core/   # embeddable library (published)
packages/cliver/        # CLI application built on cliver-core
website/                # documentation site (Astro + Starlight)
```

## Documentation

Full documentation at **[cliver-project.github.io/CLIver](https://cliver-project.github.io/CLIver/)**

## Development

```bash
git clone https://github.com/cliver-project/CLIver.git
cd CLIver

make init     # Set up dev environment
make test     # Run the workspace test suite
make lint     # Lint and format check
make format   # Auto-fix lint/format issues
make dist     # Build the cliver-core sdist + wheel

cd website && npm install && npm run dev   # Docs dev server at localhost:4321
```

## License

[Apache 2.0](LICENSE)
