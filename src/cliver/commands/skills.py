"""
Skills management commands.

List, show, create, update, and run agent skills (SKILL.md files).
"""

import asyncio
import logging
from pathlib import Path

import click
from langchain_core.messages import AIMessage, HumanMessage
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.skill_manager import SkillManager, validate_skill_name
from cliver.util import get_config_dir

logger = logging.getLogger(__name__)


@click.group(
    name="skills",
    help="Manage agent skills (SKILL.md files that guide LLM behavior for specific tasks)",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def skills(ctx, cliver: Cliver):
    """List, create, or manage skills."""
    if ctx.invoked_subcommand is None:
        _list_skills(cliver)


def _create_skill(cliver: Cliver, name: str, description: str, save_global: bool = False):
    """Create a new skill using LLM generation."""
    # Validate the skill name
    errors = validate_skill_name(name)
    if errors:
        for err in errors:
            cliver.output(f"[red]{err}[/red]")
        return

    # Determine target directory
    if save_global:
        skills_dir = get_config_dir() / "skills" / name
    else:
        skills_dir = Path.cwd() / ".cliver" / "skills" / name

    skill_file = skills_dir / "SKILL.md"
    if skill_file.exists():
        cliver.output(f"[yellow]Skill '{name}' already exists at {skill_file}[/yellow]")
        return

    # Build prompt for the LLM to generate the SKILL.md content
    prompt = _build_create_prompt(name, description)
    model = (cliver.session_options or {}).get("model") or None

    result = _llm_generate(cliver, prompt, model, f"Generating skill '{name}'...")
    if not result:
        return
    content = result

    # Write to disk
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(content, encoding="utf-8")
    cliver.output(f"Created skill at: [green]{skill_file}[/green]")
    cliver.output("You can edit the file to refine the skill content.")


def _show_skill(cliver: Cliver, name: str):
    """Display the full content of a skill."""
    manager = SkillManager()
    skill = manager.get_skill(name)
    if not skill:
        cliver.output(f"[red]Skill '{name}' not found.[/red]")
        _suggest_available(cliver, manager)
        return

    cliver.output(f"[bold]Skill: {skill.name}[/bold]")
    cliver.output(f"[dim]Source: {skill.source} — {skill.base_dir}[/dim]")
    cliver.output(f"[dim]Description: {skill.description}[/dim]")
    if skill.allowed_tools:
        cliver.output(f"[dim]Allowed tools: {', '.join(skill.allowed_tools)}[/dim]")
    cliver.output("")
    cliver.output(skill.body)


def _update_skill(cliver: Cliver, name: str, instructions: str):
    """Update an existing skill using LLM generation."""
    manager = SkillManager()
    skill = manager.get_skill(name)
    if not skill:
        cliver.output(f"[red]Skill '{name}' not found.[/red]")
        _suggest_available(cliver, manager)
        return

    skill_file = skill.base_dir / "SKILL.md"
    if not skill_file.is_file():
        cliver.output(f"[red]Cannot find SKILL.md at {skill_file}[/red]")
        return

    current_content = skill_file.read_text(encoding="utf-8")
    prompt = _build_update_prompt(name, current_content, instructions)
    model = (cliver.session_options or {}).get("model") or None

    result = _llm_generate(cliver, prompt, model, f"Updating skill '{name}'...")
    if not result:
        return
    content = result

    skill_file.write_text(content, encoding="utf-8")
    cliver.output(f"Updated skill at: [green]{skill_file}[/green]")


def _compress_if_needed(cliver, agent_core, model_config, model_name, new_input):
    """Check and compress conversation history if it exceeds context window budget."""
    from cliver.conversation_compressor import ConversationCompressor, estimate_tokens, get_context_window

    context_window = get_context_window(model_config)
    compressor = ConversationCompressor(context_window)

    if not compressor.needs_compression([], cliver.conversation_messages, new_input):
        return

    before_tokens = estimate_tokens(cliver.conversation_messages)
    llm_engine = agent_core.get_llm_engine(model_name)

    try:
        compressed = asyncio.get_event_loop().run_until_complete(
            compressor.compress(cliver.conversation_messages, llm_engine)
        )
    except RuntimeError:
        compressed = asyncio.run(compressor.compress(cliver.conversation_messages, llm_engine))

    cliver.conversation_messages = compressed
    after_tokens = estimate_tokens(cliver.conversation_messages)
    cliver.output(f"\\[Compressed conversation: ~{before_tokens} → ~{after_tokens} tokens]")


def _run_skill(cliver: Cliver, name: str, message: str = ""):
    """Activate a skill and run it through LLM inference."""
    manager = SkillManager()
    skill_obj = manager.get_skill(name)
    if not skill_obj:
        cliver.output(f"[red]Skill '{name}' not found.[/red]")
        _suggest_available(cliver, manager)
        return

    cliver.output(f"Activating skill '[green]{name}[/green]' ...")

    skill_content = manager.activate_skill(name)
    skill_system_msg = f"The user has activated the '{name}' skill. Follow these skill instructions:\n\n{skill_content}"

    def skill_appender():
        return skill_system_msg

    user_message = message if message else ""
    if not user_message:
        user_message = (
            f"I want to use the '{name}' skill. Please explain what this skill does and ask me what I'd like to do."
        )

    from cliver.cli_llm_call import LLMCallOptions, llm_call

    session_options = cliver.session_options or {}
    use_model = session_options.get("model", None)
    use_stream = session_options.get("stream", True)
    agent_core = cliver.agent_core

    cliver.record_turn("user", user_message)

    model_config = agent_core._get_llm_model(use_model)
    if model_config and cliver.conversation_messages:
        _compress_if_needed(cliver, agent_core, model_config, use_model, user_message)

    conv_history = list(cliver.conversation_messages) if cliver.conversation_messages else None
    cliver.conversation_messages.append(HumanMessage(content=user_message))

    def on_response(text: str):
        cliver.record_turn("assistant", text)
        cliver.conversation_messages.append(AIMessage(content=text))

    llm_call(
        cliver,
        LLMCallOptions(
            user_input=user_message,
            model=use_model,
            stream=use_stream,
            system_message_appender=skill_appender,
            conversation_history=conv_history,
            on_response=on_response,
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """List and manage skills — list, show, run, create, update."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "list":
        _list_skills(cliver)
    elif sub == "show":
        if not rest:
            cliver.output("[red]Missing skill name[/red]")
            return
        _show_skill(cliver, rest.strip())
    elif sub == "run":
        run_parts = rest.split(None, 1) if rest else []
        if not run_parts:
            cliver.output("[red]Missing skill name[/red]")
            cliver.output("Usage: /skills run <name> [message]")
            return
        skill_name = run_parts[0]
        message = run_parts[1] if len(run_parts) > 1 else ""
        _run_skill(cliver, skill_name, message)
    elif sub == "create":
        # Parse: create <name> <description...>
        create_parts = rest.split(None, 1) if rest else []
        if not create_parts:
            cliver.output("[red]Missing skill name and description[/red]")
            return
        skill_name = create_parts[0]
        desc = create_parts[1] if len(create_parts) > 1 else ""
        if not desc:
            cliver.output("[red]Missing description[/red]")
            return
        # Check for --global flag
        save_global = "--global" in desc
        if save_global:
            desc = desc.replace("--global", "").strip()
        _create_skill(cliver, skill_name, desc, save_global)
    elif sub == "update":
        # Parse: update <name> <instructions...>
        update_parts = rest.split(None, 1) if rest else []
        if not update_parts:
            cliver.output("[red]Missing skill name and instructions[/red]")
            return
        skill_name = update_parts[0]
        instructions = update_parts[1] if len(update_parts) > 1 else ""
        if not instructions:
            cliver.output("[red]Missing update instructions[/red]")
            return
        _update_skill(cliver, skill_name, instructions)
    elif sub in ("--help", "help"):
        cliver.output("Manage agent skills (SKILL.md files). Skills are instruction sets that guide")
        cliver.output("LLM behavior for specific tasks like brainstorming or planning.")
        cliver.output("")
        cliver.output("Usage: /skills [list|show|run|create|update] [arguments]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  list                         — List all discovered skills with name, description,")
        cliver.output("                                  and source (builtin/global/project). No parameters.")
        cliver.output("  show <name>                  — Display the full SKILL.md content of a skill.")
        cliver.output("    name  STRING (required) — Skill name. Must match a name from '/skills list'.")
        cliver.output("  run <name> [message]         — Activate a skill and run it through LLM inference.")
        cliver.output("    name     STRING (required) — Skill name to activate.")
        cliver.output("    message  STRING (optional) — Initial message to send with the skill.")
        cliver.output("             If omitted, the LLM will explain the skill and ask what to do.")
        cliver.output("  create <name> <description>  — Generate a new SKILL.md file using the LLM.")
        cliver.output("    name         STRING (required) — Skill name. Lowercase, hyphens allowed.")
        cliver.output("                   Must not already exist.")
        cliver.output("    description  STRING (required) — What the skill does (remaining words joined).")
        cliver.output("    --global     FLAG (optional) — Save to global skills directory (~/.config/cliver/skills/)")
        cliver.output("                   instead of project-local .cliver/skills/. Default: project-local.")
        cliver.output("  update <name> <instructions>  — Improve an existing skill's SKILL.md using the LLM.")
        cliver.output("    name          STRING (required) — Skill name to update. Must exist.")
        cliver.output("    instructions  STRING (required) — How to improve the skill (remaining words joined).")
        cliver.output("")
        cliver.output("Default subcommand: list (when /skills is called with no arguments)")
        cliver.output("")
        cliver.output("Examples:")
        cliver.output("  /skills                                     — list all skills")
        cliver.output("  /skills show brainstorm                     — view brainstorm skill content")
        cliver.output("  /skills run brainstorm                      — activate brainstorm, LLM asks for input")
        cliver.output("  /skills run brainstorm design a login page  — activate with an initial task")
        cliver.output("  /skills create code-review review code for quality issues")
        cliver.output("  /skills create my-skill some description --global")
        cliver.output("  /skills update brainstorm add a section about architecture decisions")
    else:
        cliver.output(f"[yellow]Unknown subcommand: /skills {sub}[/yellow]")
        cliver.output("Run '/skills help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


@skills.command(name="list", help="List all discovered skills with name, description, and source location")
@pass_cliver
def list_skills(cliver: Cliver):
    """List all discovered skills with their source locations."""
    _list_skills(cliver)


@skills.command(name="run", help="Activate a skill and run it through LLM inference with optional initial message")
@click.argument("name", type=str)
@click.argument("message", nargs=-1)
@pass_cliver
def run_skill(cliver: Cliver, name: str, message: tuple):
    """Activate a skill and run it through LLM inference.

    NAME is the skill name to activate.
    MESSAGE (optional) is the initial message to send along with the skill.
    """
    user_message = " ".join(message) if message else ""
    _run_skill(cliver, name, user_message)


@skills.command(name="create", help="Generate a new SKILL.md file using the LLM from a name and description")
@click.argument("name", type=str)
@click.argument("description", nargs=-1, required=True)
@click.option(
    "--global",
    "save_global",
    is_flag=True,
    default=False,
    help="Save to global skills directory (~/.config/cliver/skills/) instead of project-local .cliver/skills/",
)
@pass_cliver
def create_skill(cliver: Cliver, name: str, description: tuple, save_global: bool):
    """Create a new skill by asking the LLM to generate a SKILL.md file.

    NAME is the skill name (lowercase, hyphens allowed).
    DESCRIPTION is what kind of skill it is (remaining arguments joined).
    """
    desc_text = " ".join(description)
    _create_skill(cliver, name, desc_text, save_global)


@skills.command(name="show", help="Display the full SKILL.md content, metadata, and source location of a skill")
@click.argument("name", type=str)
@pass_cliver
def show_skill(cliver: Cliver, name: str):
    """Display the full content of a skill."""
    _show_skill(cliver, name)


@skills.command(name="update", help="Improve an existing SKILL.md using the LLM with natural-language instructions")
@click.argument("name", type=str)
@click.argument("instructions", nargs=-1, required=True)
@pass_cliver
def update_skill(cliver: Cliver, name: str, instructions: tuple):
    """Update an existing skill by asking the LLM to improve it.

    NAME is the skill name to update.
    INSTRUCTIONS describe how to improve the skill (remaining arguments joined).
    """
    improvement_text = " ".join(instructions)
    _update_skill(cliver, name, improvement_text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_skills(cliver: Cliver) -> None:
    """Display all discovered skills in a Rich table."""
    manager = SkillManager()
    all_skills = manager.list_skills()

    if not all_skills:
        cliver.output("No skills found.")
        cliver.output("[dim]Skills are discovered from:[/dim]")
        cliver.output("[dim]  - .cliver/skills/ (project)[/dim]")
        cliver.output(f"[dim]  - {get_config_dir() / 'skills'} (global)[/dim]")
        cliver.output("[dim]  - ~/.agents/skills/ (agent-agnostic)[/dim]")
        cliver.output("[dim]Create one with: /skills create <name> <description>[/dim]")
        return

    table = Table(title=f"Discovered Skills ({len(all_skills)})", box=box.ROUNDED)
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Description", style="white", max_width=60)
    table.add_column("Source", style="cyan", no_wrap=True)

    for s in all_skills:
        desc = s.description
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(s.name, desc, s.source)

    cliver.output(table)


def _suggest_available(cliver: Cliver, manager: SkillManager) -> None:
    """Show available skill names as a hint."""
    names = manager.get_skill_names()
    if names:
        cliver.output(f"Available skills: {', '.join(names)}")


async def _no_tools(_user_input, _tools):
    """filter_tools callback that disables all tools."""
    return []


def _llm_generate(cliver: Cliver, prompt: str, model: str, status_msg: str) -> str | None:
    """Call LLM to generate content (no tools, no streaming), with spinner and token display.

    Returns extracted SKILL.md content, or None on failure.
    """
    from cliver.cli_llm_call import LLMCallOptions, llm_call

    cliver.output(f"[bold cyan]{status_msg}[/bold cyan]")
    result = llm_call(
        cliver,
        LLMCallOptions(
            user_input=prompt,
            model=model,
            stream=False,
            tools_filter=_no_tools,
        ),
    )

    if not result.success or not result.text:
        if not result.error:
            cliver.output("[red]LLM returned no content.[/red]")
        return None

    return _extract_skill_content(result.text.strip())


_SKILL_PROMPT_RULES = "IMPORTANT: Do NOT call any skill toolsRespond with ONLY the SKILL.md file content directly. "


def _build_create_prompt(name: str, description: str) -> str:
    """Build the prompt for the LLM to generate a SKILL.md file."""
    return (
        f"Generate a SKILL.md file for a skill named '{name}'.\n"
        f"The skill is about: {description}\n\n"
        f"Requirements:\n"
        f"1. Start with YAML frontmatter between --- markers\n"
        f"2. The frontmatter MUST include:\n"
        f"   - name: {name}\n"
        f"   - description: A clear, concise description (under 200 chars)\n"
        f"3. Optionally include in frontmatter:\n"
        f"   - allowed-tools: space-delimited tool names if the skill needs specific tools\n"
        f"4. The body (after frontmatter) should contain:\n"
        f"   - Clear instructions for the LLM when this skill is activated\n"
        f"   - Step-by-step guidance or patterns to follow\n"
        f"   - Any relevant context or constraints\n\n"
        f"{_SKILL_PROMPT_RULES}"
    )


def _build_update_prompt(name: str, current_content: str, instructions: str) -> str:
    """Build the prompt for the LLM to update/improve a SKILL.md file."""
    return (
        f"Update the following SKILL.md file for skill '{name}'.\n\n"
        f"Current content:\n"
        f"```\n{current_content}\n```\n\n"
        f"Improvement instructions: {instructions}\n\n"
        f"Requirements:\n"
        f"1. Keep the same YAML frontmatter format (--- markers)\n"
        f"2. The frontmatter MUST keep name: {name}\n"
        f"3. Apply the improvement instructions to the skill content\n"
        f"4. Preserve any existing good content that isn't contradicted by the instructions\n\n"
        f"{_SKILL_PROMPT_RULES}"
    )


def _extract_skill_content(content: str) -> str:
    """Extract SKILL.md content, stripping any code fences the LLM may add."""
    # Strip markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```markdown or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    # Ensure it starts with ---
    content = content.strip()
    if not content.startswith("---"):
        content = "---\n" + content

    return content
