---
title: "Load an Agent Skill in one call"
description: "A skill is a directory with instructions the model only sees when it runs — one explicit source, one run_skill() call, and a tool allowlist that narrows the run."
pubDate: 2026-11-18
tags:
  - Skills
  - AgentCore
issue: 6
draft: true
author: Lin Gao
series: Field Notes
---

A system prompt that grows with every task eventually becomes unreadable, and the model pays for
all of it on every call. An Agent Skill moves one task's instructions into a directory and loads
them only for the run that needs them.

A skill is a `SKILL.md` with a name, a description, and a body:

```markdown title="skills/hello/SKILL.md"
---
name: hello
description: Greet the user in the language they ask for, using the house phrase list.
allowed-tools: Read
---

# Hello skill

Always greet in the language the user asks for.

1. Read `references/phrases.md` in this skill's directory.
2. Find the greeting for the requested language.
3. Reply with one sentence that starts with that greeting.
```

The frontmatter is routing metadata: name and description are all a catalogue needs. The body is
the instruction set, and it reaches the model only when the skill runs.

## One source, one call

```python title="skill_hello.py" file="samples/blog/skill_hello/skill_hello.py" region="agent"
```

`cliver-core` discovers nothing implicitly. `SkillManager(sources=[...])` takes directories the
application chooses — later sources override earlier ones by name. `run_skill("hello", prompt)`
finds the skill, appends an activation header and the body to the single system message, and runs
the normal agent loop.

The header includes the skill's location, which is how a bundled `scripts/` or `references/`
path becomes reachable. The agent here carries an empty `ExecutionPolicy` on purpose: when one is
configured, `run_skill()` pins the working directory to the skill's own directory, so
`references/phrases.md` resolves next to `SKILL.md`. Without a policy, relative paths resolve
against your process cwd.

## The allowlist is real

The skill declares `allowed-tools: Read`. The agent registers both `Read` and `Write`, but during
this run the model is advertised only `Read`; a scripted `Write` call is rejected before it
executes. A non-empty allowlist that matches zero available tools fails fast instead of silently
running unrestricted.

`run_skill()` deliberately adds none of the model-driven skill tools. The skill is already
pinned, so `FindSkills` and `ActivateSkill` have nothing to do.

## The run

```text
こんにちは！Japanese greeting for you — nice to meet you.
```

The model read the bundled phrase list through `Read`, found the Japanese greeting, and answered
in one line.

## When your code already knows the skill

`run_skill()` is the explicit path: a router, a CLI flag, or a job type picked the skill. With a
`SkillManager` configured, plain `chat()` also carries a short catalogue in the system prompt and
advertises `FindSkills` and `ActivateSkill`, so the model can choose the skill itself. Same
directory format, two dispatch modes.

The `cliver` CLI already accepts `--skill <name>` for a one-shot skill run and discovers skills
from eight sources, including project-local `.cliver`, `.claude`, and `.agents` directories. The
CLI is experimental and not on PyPI yet.

Next: [Skills](../../core-concepts/skills/) covers the format, dispatch modes, bundled files, and
the search and selector seams.
