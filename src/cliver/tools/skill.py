"""Built-in skill tool for LLM-driven skill activation."""

import logging

from cliver.skill_manager import SkillManager
from cliver.tool import tool

logger = logging.getLogger(__name__)

_skill_manager = SkillManager()


def get_skill_manager() -> SkillManager:
    return _skill_manager


@tool(name="Skill", description="Activate a skill and execute it with the skill instructions in the system prompt.")
def skill(skill_name: str, prompt: str | None = None) -> list[dict]:
    """Activate a skill to get specialized instructions and context.

    Skills provide domain-specific knowledge and guidance for tasks.
    Call with skill_name='list' to see available skills, then call
    with the specific skill name to activate it.
    """
    manager = get_skill_manager()

    if skill_name == "list":
        return [{"text": manager.format_skill_list()}]

    skill_obj = manager.get_skill(skill_name)
    if not skill_obj:
        available = manager.get_skill_names()
        msg = f"Skill '{skill_name}' not found."
        if available:
            msg += f" Available skills: {', '.join(available)}"
        return [{"text": msg}]

    if not prompt:
        return [{"text": manager.activate_skill(skill_name)}]

    # Run the skill through the LLM — create a minimal AgentCore for the call
    from cliver.agent_factory import create_agent_core, resolve_model
    from cliver.config import ConfigManager
    from cliver.util import get_config_dir

    cm = ConfigManager(get_config_dir())
    mc = resolve_model(None, cm)
    if not mc:
        return [{"text": manager.activate_skill(skill_name, prompt=prompt)}]

    agent_core = create_agent_core(model_config=mc)

    import asyncio

    system_prompt = f"# Skill: {skill_name}\n\n{skill_obj.body}"
    response = asyncio.run(
        agent_core.chat(
            user_input=prompt,
            system_prompt=system_prompt,
        )
    )
    return [{"text": response.message.text or "Skill completed with no output."}]
