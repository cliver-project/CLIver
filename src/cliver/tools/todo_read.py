"""Built-in todo_read tool for reading current plan progress."""

from cliver.tool import tool
from cliver.tools.todo_write import format_todo_summary, get_current_todos


@tool(
    name="TodoRead",
    description=(
        "Read the current plan/todo list to check progress. "
        "Returns the full todo list with status of each item. "
        "Use this to review your plan before deciding what to do next, "
        "especially after completing a step or when resuming work."
    ),
)
def todo_read() -> list[dict]:
    """Read the current plan/todo list without modifying it.

    Returns the full todo list with status of each item.
    """
    return [{"text": format_todo_summary(get_current_todos())}]
