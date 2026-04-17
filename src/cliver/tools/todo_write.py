"""Built-in todo_write tool for managing structured task lists."""

import logging
from typing import List, Literal, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Module-level todo storage (persists within a session)
_current_todos: List[dict] = []

STATUS_ICONS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}


def get_current_todos() -> List[dict]:
    """Get the current todo list (for use by other tools or the system)."""
    return list(_current_todos)


def format_todo_summary(todos: list[dict]) -> str:
    """Format a todo list into a human-readable summary.

    Shared by todo_write, todo_read, and the Re-Act loop context re-injection.
    """
    if not todos:
        return "No active plan."

    lines = ["Todo List:"]
    pending = in_progress = completed = 0

    for item in todos:
        status = item.get("status", "pending")
        icon = STATUS_ICONS.get(status, "[ ]")
        content = item.get("content", "")
        item_id = item.get("id", "?")
        lines.append(f"  {icon} ({item_id}) {content}")

        if status == "pending":
            pending += 1
        elif status == "in_progress":
            in_progress += 1
        elif status == "completed":
            completed += 1

    total = len(todos)
    lines.append(f"\nProgress: {completed}/{total} completed, {in_progress} in progress, {pending} pending")
    return "\n".join(lines)


class TodoItem(BaseModel):
    """A single todo item."""

    id: str = Field(description="Unique identifier for the todo item.")
    content: str = Field(min_length=1, description="Description of the task.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Status of the todo item: 'pending', 'in_progress', or 'completed'."
    )


class TodoWriteInput(BaseModel):
    """Input schema for the todo_write tool."""

    todos: List[TodoItem] = Field(description="The complete updated todo list.")


class TodoWriteTool(BaseTool):
    """Creates and manages a structured task list for the current session."""

    name: str = "TodoWrite"
    description: str = (
        "Creates and manages a structured task list for your current session. "
        "This helps track progress, organize complex tasks, and demonstrate thoroughness.\n\n"
        "When to use:\n"
        "1. Complex multi-step tasks requiring 3+ distinct steps\n"
        "2. User explicitly requests a todo list or work breakdown\n"
        "3. User provides multiple tasks to be done\n"
        "4. When starting work on a task - mark it as 'in_progress'\n"
        "5. After completing a task - mark it as 'completed'\n\n"
        "When NOT to use:\n"
        "- Single straightforward tasks\n"
        "- Trivial tasks completable in less than 3 steps\n"
        "- Purely conversational or informational queries\n\n"
        "Each call replaces the entire todo list. Always include all items "
        "(pending, in_progress, completed) in every call."
    )
    args_schema: Type[BaseModel] = TodoWriteInput
    tags: list = ["think", "planning", "task"]

    def _run(self, todos: List[dict]) -> str:
        global _current_todos

        # Convert dicts to TodoItem for validation, then back to dicts for storage
        validated = []
        for item in todos:
            if isinstance(item, dict):
                validated.append(item)
            elif isinstance(item, TodoItem):
                validated.append(item.model_dump())
            else:
                validated.append(dict(item))

        _current_todos = validated
        return format_todo_summary(_current_todos)


todo_write = TodoWriteTool()
