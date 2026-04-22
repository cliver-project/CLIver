"""
Skills management commands.

List, create, and activate agent skills (SKILL.md files).
"""

import logging
from pathlib import Path

import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.skill_manager import SkillManager, validate_skill_name
from cliver.util import get_config_dir

logger = logging.getLogger(__name__)


@click.group(
    name="skills",
    help="Manage agent skills",
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


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Dispatch /skills commands from string args."""
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
        cliver.output("Usage: /skills [list|create|show|update]")
        cliver.output("  list                          - List all skills")
        cliver.output("  show <name>                   - Show skill content")
        cliver.output("  create <name> <description>   - Create new skill via LLM")
        cliver.output("  update <name> <instructions>  - Update skill via LLM")
    else:
        cliver.output(f"[yellow]Unknown subcommand: /skills {sub}[/yellow]")
        cliver.output("Run '/skills help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


@skills.command(name="list", help="List all discovered skills")
@pass_cliver
def list_skills(cliver: Cliver):
    """List all discovered skills with their source locations."""
    _list_skills(cliver)


@skills.command(name="create", help="Create a new skill using LLM generation")
@click.argument("name", type=str)
@click.argument("description", nargs=-1, required=True)
@click.option(
    "--global",
    "save_global",
    is_flag=True,
    default=False,
    help="Save to global skills directory instead of project-local .cliver/skills/",
)
@pass_cliver
def create_skill(cliver: Cliver, name: str, description: tuple, save_global: bool):
    """Create a new skill by asking the LLM to generate a SKILL.md file.

    NAME is the skill name (lowercase, hyphens allowed).
    DESCRIPTION is what kind of skill it is (remaining arguments joined).
    """
    desc_text = " ".join(description)
    _create_skill(cliver, name, desc_text, save_global)


@skills.command(name="show", help="Show the full content of a skill")
@click.argument("name", type=str)
@pass_cliver
def show_skill(cliver: Cliver, name: str):
    """Display the full content of a skill."""
    _show_skill(cliver, name)


@skills.command(name="update", help="Update a skill using LLM generation")
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
