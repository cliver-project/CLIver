---
title: Skills
description: LLM-driven skill activation for specialized tasks
---

# Skills

CLIver has an LLM-driven skill system that allows the agent to discover and activate specialized capabilities during a chat session. Skills are defined as SKILL.md files with YAML frontmatter.

## How Skills Work

Unlike traditional configuration-based systems, CLIver's skills are **LLM-driven**:

1. The `skill` builtin tool is always available (core tool)
2. The LLM calls `skill('list')` to discover available skills
3. The LLM calls `skill('skill-name')` to activate a skill and receive its instructions
4. The skill content is injected into the conversation as context

This means the LLM decides when to use skills based on the user's request — no manual flags needed.

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

Skills are loaded from two locations (project-local takes priority):

1. **Project**: `.cliver/skills/` in the current working directory
2. **Global**: `~/.config/cliver/skills/`

Each skill is a subdirectory containing a `SKILL.md` file.

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
