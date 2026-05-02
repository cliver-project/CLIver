"""Built-in skill tool for LLM-driven skill activation."""

import asyncio
import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.skill_manager import SkillManager

logger = logging.getLogger(__name__)

# Module-level skill manager — lazily initialized, shared by the tool
_skill_manager = SkillManager()


def get_skill_manager() -> SkillManager:
    """Get the module-level SkillManager instance."""
    return _skill_manager


class SkillToolInput(BaseModel):
    """Input schema for the skill tool."""

    skill_name: str = Field(description="Name of the skill to activate. Use 'list' to see all available skills.")
    prompt: Optional[str] = Field(
        default=None,
        description="Initial prompt describing what you want the skill to do. "
        "The skill will execute with this prompt in a fresh context.",
    )


class SkillTool(BaseTool):
    """Activate a skill to get specialized instructions and context.

    Skills provide domain-specific knowledge and guidance for tasks.
    Call with skill_name='list' to see available skills, then call
    with the specific skill name to activate it.
    """

    name: str = "Skill"
    description: str = (
        "Activate a skill and execute it with the skill instructions in the system prompt. "
        "Skills provide expert knowledge, tool usage guidance, and step-by-step "
        "instructions for specific tasks.\n\n"
        "Usage:\n"
        "- Call with skill_name='list' to see all available skills and their descriptions\n"
        "- Call with a specific skill name and prompt to activate and run it\n\n"
        "When to use:\n"
        "- When a task matches a skill's domain (e.g., web search, code review, k8s admin)\n"
        "- When you need specialized guidance beyond your general knowledge\n"
        "- When the user's request aligns with an available skill's description"
    )
    args_schema: Type[BaseModel] = SkillToolInput
    tags: list = ["skill", "planning", "context"]

    def _run(self, skill_name: str, prompt: Optional[str] = None) -> str:
        manager = get_skill_manager()

        if skill_name == "list":
            return manager.format_skill_list()

        skill = manager.get_skill(skill_name)
        if not skill:
            available = manager.get_skill_names()
            msg = f"Skill '{skill_name}' not found."
            if available:
                msg += f" Available skills: {', '.join(available)}"
            return msg

        if not prompt:
            return manager.activate_skill(skill_name)

        from cliver.agent_profile import get_agent_core

        agent_core = get_agent_core()
        if not agent_core:
            return manager.activate_skill(skill_name, prompt=prompt)

        try:
            response = asyncio.run(agent_core.process_skill(skill_name=skill_name, user_input=prompt))
            from cliver.media_handler import extract_response_text

            return extract_response_text(response, fallback="Skill completed with no output.")
        except RuntimeError:
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(agent_core.process_skill(skill_name=skill_name, user_input=prompt))
            from cliver.media_handler import extract_response_text

            return extract_response_text(response, fallback="Skill completed with no output.")


skill = SkillTool()
