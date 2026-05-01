---
title: Skills
description: LLM-driven skill activation for specialized tasks
---

# Skills

CLIver has an LLM-driven skill system that allows the agent to discover and activate specialized capabilities during a chat session. Skills are defined as SKILL.md files with YAML frontmatter.

## How Skills Work

Skills can be activated in two ways:

### LLM-driven activation (automatic)

1. The `skill` builtin tool is always available (core tool)
2. The LLM calls `skill('list')` to discover available skills
3. The LLM calls `skill('skill-name')` to activate a skill and receive its instructions
4. The skill content is injected into the conversation as context

This means the LLM decides when to use skills based on the user's request — no manual flags needed.

### User-driven activation (manual)

Use the `/skills run` command to explicitly activate a skill and run it through LLM inference:

```
/skills run <name> [message]
```

- `/skills run brainstorm` — activate the brainstorm skill; the LLM will explain it and ask for input
- `/skills run brainstorm design a login page` — activate with an initial task message

The skill content is injected as a system message for the LLM call, so the LLM follows the skill's guidance.

## Creating a Skill

A skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```
my-skill/
└── SKILL.md
```

### SKILL.md Format

```markdown
---
name: code-review
description: Performs thorough code reviews with security and quality analysis
tags:
  - code
  - review
  - security
---

# Code Review Skill

When reviewing code, follow these steps:

1. Check for security vulnerabilities (SQL injection, XSS, etc.)
2. Review code quality and adherence to best practices
3. Identify potential performance issues
4. Suggest improvements with examples

## Output Format

Provide your review in sections:
- **Security**: List any security concerns
- **Quality**: Code quality observations
- **Performance**: Performance considerations
- **Suggestions**: Actionable improvements
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier |
| `description` | Yes | Brief description (shown in skill list) |
| `tags` | No | Keywords for discovery |

## Skill Discovery Paths

Skills are loaded from multiple locations (later entries override earlier ones):

1. **Builtin**: shipped with the CLIver package (`src/cliver/skills/`)
2. **Global (CLIver)**: `~/.config/cliver/skills/`
3. **Global (agent-agnostic)**: `~/.agents/skills/`
4. **Project (CLIver)**: `.cliver/skills/`
5. **Project (agent-agnostic)**: `.agent/skills/`
6. **Project (Claude Code compat)**: `.claude/skills/`
7. **Project (Gemini compat)**: `.gemini/skills/`
8. **Project (Qwen Code compat)**: `.qwen/skills/`

Each skill is a subdirectory containing a `SKILL.md` file.

## Managing Skills

Use the `/skills` command to manage skills interactively:

| Subcommand | Description |
|---|---|
| `/skills` or `/skills list` | List all discovered skills with name, description, and source |
| `/skills show <name>` | Display the full SKILL.md content of a skill |
| `/skills run <name> [message]` | Activate a skill and run it through LLM inference |
| `/skills create <name> <description>` | Generate a new SKILL.md file using the LLM |
| `/skills update <name> <instructions>` | Improve an existing skill using the LLM |

## Workflow Integration

Skills can be activated within workflow LLM steps using the `skills` field:

```yaml
name: review-workflow
steps:
  - id: review
    type: llm
    name: Code Review
    prompt: "Review the code in {{ inputs.file_path }}"
    skills:
      - code-review
    outputs: [review_result]
```

When a step declares skills, the skill content is deterministically injected into the LLM context (not probabilistic activation).

## Example Skills

### Greeting Skill

```markdown
---
name: greeting
description: Greets users in a friendly and personalized way
tags:
  - greeting
  - welcome
---

# Greeting Skill

Greet the user warmly. If you know their name (from identity or memory), use it.
Mention the current time of day if possible.
```

### Research Skill

```markdown
---
name: research
description: Structured research with web search and analysis
tags:
  - research
  - web
  - analysis
---

# Research Skill

When conducting research:

1. Use `web_search` to find relevant sources
2. Use `web_fetch` to read promising results
3. Synthesize findings into a structured summary
4. Cite sources with URLs
```
