"""Built-in skill tool — returns skill body for the LLM to follow inline."""

import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.skill_manager import SkillManager

logger = logging.getLogger(__name__)

_skill_manager = SkillManager()


def get_skill_manager() -> SkillManager:
    """Get the module-level SkillManager instance."""
    return _skill_manager


class SkillToolInput(BaseModel):
    """Input schema for the skill tool."""

    skill_name: str = Field(description="Name of the skill to activate. Use 'list' to see all available skills.")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional context describing what you want the skill to help with.",
    )


class SkillTool(BaseTool):
    """Activate a skill by returning its instructions for you to follow.

    Skills provide domain-specific knowledge and step-by-step guidance.
    Call with skill_name='list' to see available skills, then call
    with a specific name to get its instructions.
    """

    name: str = "Skill"
    description: str = (
        "Activate a skill to get specialized instructions and guidance. "
        "The skill body is returned as text — read and follow the instructions "
        "in the current conversation.\n\n"
        "Usage:\n"
        "- Call with skill_name='list' to see all available skills\n"
        "- Call with a skill name to get its instructions\n"
        "- Optionally pass a prompt for additional context"
    )
    args_schema: Type[BaseModel] = SkillToolInput
    tags: list = ["skill", "planning", "context"]

    def _run(self, skill_name: str, prompt: Optional[str] = None) -> str:
        manager = get_skill_manager()

        if skill_name == "list":
            return manager.format_skill_list()

        return manager.activate_skill(skill_name, prompt=prompt or "")


skill = SkillTool()
