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
    prompt = _build_create_prompt(name, desc_text)
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


@skills.command(name="show", help="Show the full content of a skill")
@click.argument("name", type=str)
@pass_cliver
def show_skill(cliver: Cliver, name: str):
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


@skills.command(name="update", help="Update a skill using LLM generation")
@click.argument("name", type=str)
@click.argument("instructions", nargs=-1, required=True)
@pass_cliver
def update_skill(cliver: Cliver, name: str, instructions: tuple):
    """Update an existing skill by asking the LLM to improve it.

    NAME is the skill name to update.
    INSTRUCTIONS describe how to improve the skill (remaining arguments joined).
    """
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
    improvement_text = " ".join(instructions)
    prompt = _build_update_prompt(name, current_content, improvement_text)
    model = (cliver.session_options or {}).get("model") or None

    result = _llm_generate(cliver, prompt, model, f"Updating skill '{name}'...")
    if not result:
        return
    content = result

    skill_file.write_text(content, encoding="utf-8")
    cliver.output(f"Updated skill at: [green]{skill_file}[/green]")


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
