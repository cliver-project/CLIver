"""Built-in skill tool for LLM-driven skill activation."""

import logging
from typing import Type

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

    skill_name: str = Field(description=("Name of the skill to activate. Use 'list' to see all available skills."))


class SkillTool(BaseTool):
    """Activate a skill to get specialized instructions and context.

    Skills provide domain-specific knowledge and guidance for tasks.
    Call with skill_name='list' to see available skills, then call
    with the specific skill name to activate it.
    """

    name: str = "Skill"
    description: str = (
        "Activate a skill to get specialized instructions and context for a domain. "
        "Skills provide expert knowledge, tool usage guidance, and step-by-step "
        "instructions for specific tasks.\n\n"
        "Usage:\n"
        "- Call with skill_name='list' to see all available skills and their descriptions\n"
        "- Call with a specific skill name to activate it and receive its instructions\n\n"
        "When to use:\n"
        "- When a task matches a skill's domain (e.g., web search, code review, k8s admin)\n"
        "- When you need specialized guidance beyond your general knowledge\n"
        "- When the user's request aligns with an available skill's description"
    )
    args_schema: Type[BaseModel] = SkillToolInput
    tags: list = ["skill", "planning", "context"]

    def _run(self, skill_name: str) -> str:
        manager = get_skill_manager()

        if skill_name == "list":
            return manager.format_skill_list()

        return manager.activate_skill(skill_name)


skill = SkillTool()
