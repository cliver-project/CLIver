<div align="center">
  <img src="docs/images/icon.png" alt="CLIver Logo" width="200"/>
  
  # CLIver
  
  **General-purpose AI agent for your terminal — safe, controlled, and adaptable to any domain.**
  
  [![PyPI version](https://img.shields.io/pypi/v/cliver.svg)](https://pypi.org/project/cliver/)
  [![Python versions](https://img.shields.io/pypi/pyversions/cliver.svg)](https://pypi.org/project/cliver/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://cliver-project.github.io/CLIver/)
</div>

---

## Why CLIver?

- **Model-agnostic** — OpenAI-compatible APIs (Qwen, DeepSeek, Cerebras, etc.), Ollama, vLLM, or any provider. You choose the model, not the framework.
- **General-purpose** — Not just for coding. DevOps, research, writing, data analysis, or anything else you need. Customizable system prompts, skills, and workflows adapt to your domain.
- **Safe by default** — Layered permission system governs every tool execution. Default, auto-edit, and YOLO modes give you the right balance of control and automation.
- **Embeddable** — `AgentCore` Python API lets you integrate CLIver's agent capabilities into your own applications and scripts.
- **Gateway mode** — Deploy as a bot service with Telegram, Discord, Slack, or Feishu adapters. One agent, multiple frontends.

---

## Quick Start

```bash
# Install via pip
pip install cliver

# Or run with Docker
docker run --rm -it --user $UID:0 -v ~/.cliver:/home/cliver/.cliver \
  -e OPENAI_API_KEY ghcr.io/cliver-project/cliver

# Start chatting
cliver -p "What time is it in Beijing and London?"

# Interactive mode
cliver
```

---

## Features

### Multi-Provider LLM

Connect to any OpenAI-compatible API, Ollama, or vLLM endpoint. Configure multiple providers and switch between them on the fly. No vendor lock-in.

```yaml
# ~/.cliver/config.yaml
providers:
  - name: openai
    type: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4o
  
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    default_model: qwen2.5-coder:32b
  
  - name: deepseek
    type: openai
    base_url: https://api.deepseek.com
    api_key: ${DEEPSEEK_API_KEY}
    default_model: deepseek-chat
```

### MCP Integration

Connect to any [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server for extended tool capabilities. File systems, databases, APIs, and more — all through a standardized protocol.

```yaml
# ~/.cliver/mcp_config.yaml
mcpServers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
  
  postgres:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
```

### Skills

LLM-driven skill activation following the [Agent Skills](https://agentskills.io) specification. Skills are auto-discovered from `~/.cliver/skills/` and activated based on task context.

```bash
# List available skills
cliver skills list

# Add a custom skill
cat > ~/.cliver/skills/my-skill.yaml <<EOF
name: my-skill
description: Custom skill for domain-specific tasks
system_prompt: |
  You are an expert in my specific domain...
tools:
  - name: custom_tool
    ...
EOF
```

### Workflows

LangGraph-powered multi-step workflow engine with subagent isolation and pause/resume support. Complex tasks stay on track without scattered, unpredictable behavior.

```python
# Workflow definition (Python)
from cliver.workflow import Workflow, WorkflowStep

workflow = Workflow(
    name="deploy-pipeline",
    steps=[
        WorkflowStep(name="build", agent="builder"),
        WorkflowStep(name="test", agent="tester", depends_on=["build"]),
        WorkflowStep(name="deploy", agent="deployer", depends_on=["test"]),
    ]
)
```

### Memory & Identity

Persistent knowledge and `CliverProfile` management across sessions. CLIver remembers your preferences, project context, and domain-specific information.

```bash
# Set your profile
cliver profile set name "Leo Gao"
cliver profile set role "Platform Engineer"
cliver profile set preferences.style "concise"

# Add persistent knowledge
cliver memory add "Production cluster endpoint: https://api.prod.example.com"
```

### Permission System

Layered tool permission system with three modes:

- **default** — Explicit confirmation for every tool execution
- **auto-edit** — Auto-approve file edits and reads, confirm everything else
- **yolo** — Auto-approve all tool executions (use with caution)

```yaml
# ~/.cliver/config.yaml
permission_mode: auto-edit

# Per-tool overrides
permissions:
  file_read: allow
  file_write: allow
  bash_exec: confirm
  network_request: deny
```

### Gateway Mode

Deploy CLIver as a bot service with adapters for Telegram, Discord, Slack, or Feishu. One agent, multiple frontends.

```bash
# Install gateway dependencies
pip install cliver[telegram]

# Run Telegram bot
export TELEGRAM_BOT_TOKEN=your_token
cliver gateway telegram

# Or use Docker
docker run -e TELEGRAM_BOT_TOKEN -e OPENAI_API_KEY \
  ghcr.io/cliver-project/cliver:latest gateway telegram
```

### Embeddable Python API

Use `AgentCore` to integrate CLIver's agent capabilities into your own applications.

```python
from cliver import AgentCore

# Initialize agent
agent = AgentCore(
    provider="openai",
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
)

# Run a task
response = agent.run("Analyze this data and generate a report")
print(response.content)

# Stream responses
for chunk in agent.stream("What's the weather in Tokyo?"):
    print(chunk, end="", flush=True)
```

---

## Documentation

Full documentation is available at [https://cliver-project.github.io/CLIver/](https://cliver-project.github.io/CLIver/)

| Topic | Description |
|-------|-------------|
| [Installation](https://cliver-project.github.io/CLIver/installation/) | Installation methods, requirements, and setup |
| [Configuration](https://cliver-project.github.io/CLIver/configuration/) | Configure providers, models, permissions, and behavior |
| [Chat Usage](https://cliver-project.github.io/CLIver/chat/) | Interactive mode, one-shot commands, and session management |
| [Skills](https://cliver-project.github.io/CLIver/skills/) | Built-in skills, custom skills, and Agent Skills spec |
| [Memory & Identity](https://cliver-project.github.io/CLIver/memory-identity/) | Profile management and persistent knowledge |
| [Permissions](https://cliver-project.github.io/CLIver/permissions/) | Permission modes, tool allowlists, and safety controls |
| [Workflows](https://cliver-project.github.io/CLIver/workflow/) | LangGraph workflows, subagents, and task orchestration |
| [Extensibility](https://cliver-project.github.io/CLIver/extensibility/) | Custom tools, plugins, and API integration |

---

## Development

CLIver welcomes contributions. To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/cliver-project/CLIver.git
cd CLIver

# Set up dev environment
make init

# Run tests
make test

# Lint and format check
make lint

# Auto-fix lint and formatting
make format
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

CLIver is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built by <a href="https://github.com/leogao">Leo Gao</a> | 16-year Red Hat veteran</sub>
</div>
