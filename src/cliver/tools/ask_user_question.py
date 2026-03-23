"""Built-in ask_user_question tool for interactive user confirmation and input."""

import logging
from typing import List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_input_fn

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

    name: str = "ask_user_question"
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
        prompt = get_input_fn()
        try:
            print(f"\n{'=' * 60}")
            print(f"  Question: {question}")
            print(f"{'=' * 60}")

            if options:
                # Display numbered options
                for i, opt in enumerate(options, 1):
                    label = opt.get("label", "") if isinstance(opt, dict) else opt.label
                    desc = opt.get("description", "") if isinstance(opt, dict) else opt.description
                    print(f"  [{i}] {label}")
                    if desc:
                        print(f"      {desc}")
                print("  [0] Other (type custom response)")
                print()

                response = prompt("  Your choice (number or text): ").strip()

                # Check if it's a number selection
                try:
                    choice = int(response)
                    if 1 <= choice <= len(options):
                        opt = options[choice - 1]
                        selected = opt.get("label", "") if isinstance(opt, dict) else opt.label
                        print(f"  Selected: {selected}")
                        return f"User selected: {selected}"
                    elif choice == 0:
                        custom = prompt("  Your response: ").strip()
                        return f"User response: {custom}"
                except ValueError:
                    pass

                # Treat as free text
                return f"User response: {response}"
            else:
                response = prompt("  Your response: ").strip()
                return f"User response: {response}"
        except (EOFError, KeyboardInterrupt):
            return "User cancelled the question."
        except Exception as e:
            return f"Error asking user: {str(e)}"


# Module-level instance for tool registry
ask_user_question = AskUserQuestionTool()
