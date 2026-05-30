"""System prompt builder — stateless section helpers."""

import os
from datetime import datetime, timezone


def build(
    *,
    agent_name: str = "CLIver",
    available_tools: set[str] | None = None,
    enabled_skills: set[str] | None = None,
    models: dict | None = None,
    agents: dict | None = None,
    current_model: str | None = None,
    current_provider: str | None = None,
) -> str:
    sections = [
        _section_identity(agent_name),
        _section_self_awareness(available_tools, models, agents, current_model, current_provider),
    ]
    sections.append(_section_tool_usage())
    sections.append(_section_interaction_guidelines(available_tools, enabled_skills))
    sections.append(_section_response_format())
    return "\n\n".join(sections)


def _section_identity(agent_name: str) -> str:
    cwd = os.getcwd()
    try:
        from cliver.util import format_datetime, get_effective_timezone

        tz = get_effective_timezone()
        tz_name = str(tz)
        now_aware = datetime.now(timezone.utc).astimezone(tz)
        utc_offset = now_aware.strftime("%z")
        now_local = format_datetime(fmt="%Y-%m-%d %H:%M:%S")
    except Exception:
        tz_name = "unknown"
        utc_offset = ""
        now_local = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return (
        "# Identity\n\n"
        f"You are **{agent_name}**, a general-purpose AI agent. "
        "You help users accomplish a wide variety of tasks.\n\n"
        "## Environment\n\n"
        f"- Working directory: `{cwd}`\n"
        f"- Local time: {now_local}\n"
        f"- Timezone: {tz_name} (UTC{utc_offset})\n\n"
        "- All file operations should be relative to this directory "
        "unless the user explicitly specifies an absolute path.\n"
        "- Do NOT list or access `/`, `/etc`, `/usr`, or other system directories "
        "unless the user specifically asks for it."
    )


def _section_self_awareness(
    available_tools: set[str] | None = None,
    models: dict | None = None,
    agents: dict | None = None,
    current_model: str | None = None,
    current_provider: str | None = None,
) -> str:
    def _has(*names: str) -> bool:
        return available_tools is None or bool(available_tools & set(names))

    from cliver.util import get_config_dir

    config_dir = get_config_dir()
    lines = [
        "# Self-Awareness\n",
        "You are powered by CLIver, a configurable AI agent platform.",
    ]

    # ── Model inventory ──────────────────────────────────────
    if models:
        by_cat: dict[str, list] = {}
        for name, mc in models.items():
            cat = getattr(mc, "category", "text") or "text"
            by_cat.setdefault(cat, []).append(name)

        active_marker = " **(active)**"
        lines.append("\n## Configured Models\n")
        for cat in ("text", "image", "audio", "video"):
            items = by_cat.get(cat, [])
            if not items:
                continue
            lines.append(f"\n### {cat.title()}")
            for name in items:
                marker = active_marker if name == current_model else ""
                provider = getattr(models.get(name, None), "provider", "")
                provider_note = f" ({provider})" if provider else ""
                lines.append(f"- **`{name}`**{provider_note}{marker}")
    elif current_model:
        provider_note = f" via the **{current_provider}** protocol" if current_provider else ""
        lines.append(f"\nYou are running as the **`{current_model}`** model{provider_note}.")

    # ── Agent profiles ───────────────────────────────────────
    if agents:
        lines.append("\n## Agent Profiles\n")
        for aname, acfg in agents.items():
            role = getattr(acfg, "role", None) or getattr(acfg, "description", None) or ""
            lines.append(f"- **{aname}**: {role}" if role else f"- **{aname}**")

    # ── Key files ────────────────────────────────────────────
    lines.append("\n## Key files you can read and edit\n")
    lines.append(f"- Config: `{config_dir}/config.yaml`")
    if _has("Identity"):
        lines.append(f"- Identity: `{config_dir}/identity.md`")
    if _has("MemoryRead", "MemoryWrite"):
        lines.append(f"- Memory: `{config_dir}/memory.md`")
    if _has("Skill"):
        lines.append(f"- Skills: `.cliver/skills/` (project) or `{config_dir}/skills/` (global)")
    lines.append(f"- Tasks: `{config_dir}/tasks/`")
    cmds = "/model, /config, /gateway, /mcp, /skills, /identity, /profile, /cost, /provider, /task"
    lines.append(f"\n## Commands\n\n{cmds}")
    return "\n".join(lines)


def _section_tool_usage() -> str:
    return (
        "# Tool Usage\n\n"
        "You have access to tools that extend your capabilities.\n\n"
        "## How to call tools\n\n"
        "- Use the structured tool-calling mechanism provided by the model API.\n"
        "- Use the exact tool name as given — do not invent or guess.\n"
        "- Supply arguments that match the parameter schema.\n"
        "- You may call multiple tools in a single response when calls are independent.\n\n"
        "## Iterative tool use\n\n"
        "After each tool call you will receive the result. You may make additional "
        "tool calls based on the results until you have enough information.\n\n"
        "If you already have enough information, respond directly."
    )


def _section_interaction_guidelines(
    available_tools: set[str] | None = None,
    enabled_skills: set[str] | None = None,
) -> str:
    def _has(*names: str) -> bool:
        return available_tools is None or bool(available_tools & set(names))

    parts = ["# Interaction Guidelines\n"]
    parts.append(
        "## Asking the user\n\n"
        "When you need to clarify or gather information, ask directly.\n"
        "Use structured format when the UI supports it."
    )
    if _has("Skill"):
        parts.append("## Skills\n")
        try:
            from cliver.tools.skill import get_skill_manager

            skills = get_skill_manager().list_skills()
            if enabled_skills is not None:
                skills = [s for s in skills if s.name in enabled_skills]
            else:
                skills = [s for s in skills if s.name in {"brainstorm", "write-plan", "execute-plan"}]
            if skills:
                parts.append("Available skills — call `Skill(skill_name='<name>')`:\n")
                for s in skills:
                    desc = s.description[:120] + "..." if len(s.description) > 120 else s.description
                    parts.append(f"- **{s.name}**: {desc}")
        except Exception:
            pass
        parts.append("\nActivate ONE skill at a time.")
    parts.append(
        "## Planning\n"
        "1. Simple (1-2 steps): Respond directly.\n"
        "2. Medium (3-5 steps): Use TodoWrite/TodoRead.\n"
        "3. Complex: Use the planning pipeline (Skill + brainstorm/write-plan/execute-plan)."
    )
    parts.append("## Error handling\n\nIf a tool call fails, analyse the error and try an alternative approach.")
    parts.append(
        "## Accuracy\n\n"
        "Do not fabricate or guess facts, URLs, file paths, API names, version numbers, "
        "command flags, or configuration syntax. If you do not know something, say so "
        "rather than inventing a plausible answer. Prefer reading files or running "
        "discovery commands to verify state before making claims about it. "
        "When you are uncertain, state your confidence level explicitly."
    )
    parts.append("## Security\n\nNever read, display, or log credentials, API keys, private keys, or secrets.")
    return "\n\n".join(parts)


def _section_response_format() -> str:
    return (
        "# Response Format\n\n"
        "- Respond in Markdown format.\n"
        "- Be concise and direct.\n"
        "- When presenting structured data, use tables, lists, or code blocks."
    )
