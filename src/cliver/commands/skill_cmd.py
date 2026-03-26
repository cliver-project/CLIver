"""
Skill activation command.

/skill <name> activates a skill and injects its content into the chat.
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.skill_manager import SkillManager


@click.command(name="skill", help="Activate a skill by name")
@click.argument("name", type=str)
@pass_cliver
def skill_cmd(cliver: Cliver, name: str):
    """Activate a skill and inject its content into the conversation.

    /skill <name> looks up the skill, then routes its instructions
    through chat so the LLM follows the skill guidance.
    """
    manager = SkillManager()
    skill_obj = manager.get_skill(name)
    if not skill_obj:
        cliver.output(f"[red]Skill '{name}' not found.[/red]")
        names = manager.get_skill_names()
        if names:
            cliver.output(f"Available skills: {', '.join(names)}")
        return

    cliver.output(f"Activating skill '[green]{name}[/green]' ...")

    # Inject the skill content as a system message appender and route through chat
    skill_content = manager.activate_skill(name)
    prompt = (
        f"I want to use the '{name}' skill. Here are the skill instructions:\n\n"
        f"{skill_content}\n\n"
        f"Please follow the skill instructions above and help me with this skill. "
        f"Start by explaining what this skill does and ask me what I'd like to do."
    )
    cliver.call_cmd(f'chat "{_escape_for_shell(prompt)}"')


def _escape_for_shell(text: str) -> str:
    """Escape text for use inside double quotes in shell_split."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
