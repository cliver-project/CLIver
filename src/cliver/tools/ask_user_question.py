"""Built-in ask_user_question tool for interactive user confirmation and input."""

import logging
from typing import List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QuestionOption(BaseModel):
    """An option for a question."""

    label: str = Field(description="The display text for this option. Should be concise (1-5 words).")
    description: str = Field(description="Explanation of what this option means or what will happen if chosen.")


class Question(BaseModel):
    """A question to ask the user."""

    question: str = Field(
        description="The complete question to ask the user. Should be clear, specific, and end with a question mark."
    )
    options: Optional[List[QuestionOption]] = Field(
        default=None,
        description="Optional: Available choices for this question (2-4 options). "
        "If not provided, the user can type a free-form response.",
    )


class AskUserQuestionInput(BaseModel):
    """Input schema for ask_user_question tool."""

    question: str = Field(
        description="The complete question to ask the user. Should be clear, specific, and end with a question mark."
    )
    options: Optional[List[QuestionOption]] = Field(
        default=None,
        description="Optional: Available choices for this question (2-4 options). "
        "If not provided, the user can type a free-form response.",
    )


class AskUserQuestionTool(BaseTool):
    """Asks the user a question and returns their response."""

    name: str = "Ask"
    description: str = (
        "Use this tool when you need to ask the user a question during execution. "
        "This allows you to:\n"
        "1. Gather user preferences or requirements\n"
        "2. Clarify ambiguous instructions\n"
        "3. Get decisions on implementation choices\n"
        "4. Request confirmation before irreversible actions\n\n"
        "Usage notes:\n"
        "- If you provide options, the user can select one or type a custom response\n"
        "- If you recommend a specific option, make that the first option and add '(Recommended)' to the label\n"
        "- Keep questions clear and specific"
    )
    args_schema: Type[BaseModel] = AskUserQuestionInput
    tags: list = ["think", "interact", "confirm"]

    def _run(self, question: str, options: Optional[List[dict]] = None) -> str:
        try:
            from cliver.agent_profile import get_cli_instance
            from cliver.cli_dialog import show_dialog

            cliver_inst = get_cli_instance()
            console = cliver_inst.console if cliver_inst else None

            # Fallback console for non-TUI mode
            if console is None:
                from rich.console import Console

                console = Console()

            # Build numbered options if provided
            numbered_opts = None
            if options:
                numbered_opts = []
                for opt in options:
                    label = opt.get("label", "") if isinstance(opt, dict) else opt.label
                    desc = opt.get("description", "") if isinstance(opt, dict) else opt.description
                    numbered_opts.append((label, desc))

            result = show_dialog(
                console=console,
                cliver_inst=cliver_inst,
                title="[bold cyan]❓ Question[/bold cyan]",
                body_text=question,
                numbered_options=numbered_opts,
                border_style="cyan",
                default_on_cancel="User cancelled the question.",
            )

            if numbered_opts and result.startswith("User "):
                return result
            if not result or result == "User cancelled the question.":
                return "User cancelled the question."
            return f"User response: {result}"

        except Exception as e:
            return f"Error asking user: {str(e)}"


# Module-level instance for tool registry
ask_user_question = AskUserQuestionTool()
