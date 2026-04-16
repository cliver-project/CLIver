---
title: Memory & Identity
description: Agent memory, identity profiles, and multi-agent isolation
---

# Memory & Identity

CLIver provides persistent memory and identity systems that allow the agent to remember information across conversations and maintain a personalized profile.

## Agent Profile

Each agent has an isolated profile managed by the `AgentProfile` class. Profiles are scoped by `agent_name` (configured in `config.yaml`):

```
~/.config/cliver/agents/{agent_name}/
├── identity.md     # Agent identity profile
├── memory.md       # Agent-scoped memory
├── sessions/       # Conversation history
└── tasks/          # Task definitions
```

A global memory file also exists at `~/.config/cliver/memory.md` for cross-agent information.

### Multi-Agent Isolation

Different agent names get fully isolated profiles. This allows running multiple agent personas with separate memory, identity, and session histories:

```yaml
# In config.yaml
agent_name: CodeHelper    # All resources scoped to this name
```

## Memory

The memory system allows the LLM to store and retrieve information across conversations. It is **LLM-driven** — the agent decides when to read or write memory based on context.

### Builtin Tools

- **`memory_read`**: Reads the current memory content
- **`memory_write`**: Writes to memory with two modes:
  - **Append**: Add new information with an optional comment
  - **Rewrite**: Replace the entire memory content

### How It Works

1. Memory content is automatically injected into the system prompt at conversation start
2. During conversation, the LLM can call `memory_write` to persist important information
3. On the next conversation, the stored memory is available in the system prompt

### Example Memory Content

```markdown
## Project Preferences
- User prefers pytest over unittest
- Always use type hints in Python code
- Project uses ruff for formatting

## Known Context
- Main API endpoint: https://api.example.com
- Database: PostgreSQL 15
- Deployment: Kubernetes on AWS
```

## Identity

The identity system maintains a living markdown document that describes the agent's persona, the user's preferences, and the relationship between them.

### Builtin Tool

- **`identity_update`**: Rewrites the identity profile as a complete markdown document

### CLI Command

```bash
# View current identity
cliver identity

# Guided conversation to build/update profile
cliver identity chat

# Clear identity profile
cliver identity clear
```

### How It Works

1. Identity content is injected into the system prompt at conversation start
2. The LLM can update identity via `identity_update` when it learns about the user
3. Identity is always a full rewrite (not append) — it's a living document

### Example Identity Content

```markdown
# Agent Identity

## User Profile
- Name: Alice
- Role: Backend developer
- Primary languages: Python, Go
- Preferred communication: Concise, technical

## Agent Persona
- Name: CLIver
- Style: Professional but friendly
- Focus: Code quality and security
```

## Graceful Degradation

When no agent profile is set (e.g., when using AgentCore as a library without configuring a profile), the memory and identity tools return "not available" messages instead of failing. This ensures the agent continues to function without these optional features.

## API-Level Usage

When using CLIver as a Python library, set up the agent profile before creating the AgentCore:

```python
from cliver.agent_profile import AgentProfile, set_current_profile

profile = AgentProfile(
    agent_name="MyAgent",
    config_dir=Path("~/.config/cliver").expanduser(),
)
set_current_profile(profile)
```

The profile is accessed via `get_current_profile()` by all builtin tools that need it.
