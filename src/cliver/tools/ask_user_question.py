"""Built-in ask_user_question tool for interactive user confirmation and input."""

import logging
from typing import Optional

from cliver.tool import tool

logger = logging.getLogger(__name__)


def _parse_choice_with_context(text: str) -> tuple[int | None, str]:
    """Parse user input like '1', '0, my custom text', or '2 extra context'.

    Returns (choice_number, extra_text). If the input doesn't start with
    a digit, returns (None, '').
    """
    text = text.strip()
    if not text or not text[0].isdigit():
        return None, ""

    import re

    m = re.match(r"^(\d+)(?:\s*[,，]\s*|\s+)(.*)", text)
    if m:
        return int(m.group(1)), m.group(2).strip()
    # Bare number with no extra text
    try:
        return int(text), ""
    except ValueError:
        return None, ""


@tool(
    name="Ask",
    description=(
        "Use this tool when you need to ask the user a question during execution. "
        "This allows you to:\n"
        "1. Gather user preferences or requirements\n"
        "2. Clarify ambiguous instructions\n"
        "3. Get decisions on implementation choices\n"
        "4. Request confirmation before irreversible actions\n\n"
        "Usage notes:\n"
        "- If you provide options, the user can select one or type a custom response\n"
        "- If you recommend a specific option, make that the first option and add "
        "'(Recommended)' to the label\n"
        "- Keep questions clear and specific"
    ),
)
def ask_user_question(
    question: str,
    options: Optional[list[dict]] = None,
) -> list[dict]:
    """Asks the user a question and returns their response.

    Gather user preferences, clarify ambiguous instructions, get decisions
    on implementation choices, or request confirmation before irreversible actions.

    Args:
        question: The complete question to ask the user. Should be clear, specific,
            and end with a question mark.
        options: Optional available choices for this question (2-4 options).
            Each option should be a dict with keys:
            - label (str): The display text for this option (1-5 words).
            - description (str): Explanation of what this option means or will happen.
            If not provided, the user can type a free-form response.
    """
    try:
        from cliver.agent_profile import get_cli_instance

        cliver_inst = get_cli_instance()

        # Build question text with numbered options
        prompt = question
        if options:
            prompt += "\n"
            for i, opt in enumerate(options, 1):
                label = opt.get("label", "") if isinstance(opt, dict) else opt.label
                desc = opt.get("description", "") if isinstance(opt, dict) else opt.description
                prompt += f"\n  [{i}] {label}"
                if desc:
                    prompt += f" - {desc}"
            prompt += "\n  [0] Other (type custom response)"

        if cliver_inst:
            from rich.panel import Panel

            cliver_inst.console.print()
            cliver_inst.console.print(
                Panel(
                    prompt,
                    title="[bold cyan]Question[/bold cyan]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
            result = cliver_inst.ui.ask_input("  > ")
        else:
            print(f"\nQuestion: {prompt}")
            try:
                result = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                return [{"text": "User cancelled the question."}]

        if not result:
            return [{"text": "User cancelled the question."}]

        # Handle numbered options, with optional extra context after comma/space
        if options and result:
            choice_num, extra = _parse_choice_with_context(result)
            if choice_num is not None:
                if 1 <= choice_num <= len(options):
                    opt = options[choice_num - 1]
                    label = opt.get("label", "") if isinstance(opt, dict) else opt.label
                    if extra:
                        return [{"text": f"User selected: {label}. Additional context: {extra}"}]
                    return [{"text": f"User selected: {label}"}]
                elif choice_num == 0:
                    if extra:
                        return [{"text": f"User response: {extra}"}]
                    if cliver_inst:
                        custom = cliver_inst.ui.ask_input("  Your response: ")
                    else:
                        custom = input("  Your response: ").strip()
                    return [{"text": f"User response: {custom}"}]
            return [{"text": f"User response: {result}"}]

        return [{"text": f"User response: {result}"}]

    except Exception as e:
        return [{"error": f"Error asking user: {str(e)}"}]
