"""Built-in todo_write tool for managing structured task lists."""

import logging

from cliver_core.run_state import current_tool_run_state
from cliver_core.tool import tool

logger = logging.getLogger(__name__)

# Fallback for direct tool calls outside an AgentCore run.  AgentCore always
# provides a per-run ToolRunState, so this list is never shared between runs.
_current_todos: list[dict] = []

_TODOS_KEY = "todos"

STATUS_ICONS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}


def _run_todos() -> list[dict] | None:
    """Return this run's todo list, or ``None`` outside an AgentCore run."""
    state = current_tool_run_state()
    if state is None:
        return None
    todos = state.data.get(_TODOS_KEY)
    if not isinstance(todos, list):
        todos = []
        state.data[_TODOS_KEY] = todos
    return todos


def get_current_todos() -> list[dict]:
    """Get the current todo list for this run (or the direct-call fallback)."""
    todos = _run_todos()
    if todos is None:
        return list(_current_todos)
    return list(todos)


def _store_todos(validated: list[dict]) -> list[dict]:
    todos = _run_todos()
    if todos is None:
        global _current_todos
        _current_todos = validated
        return _current_todos
    # Mutate in place: the worker thread's context copy shares this object,
    # but rebinding the context variable inside the worker would be lost.
    todos[:] = validated
    return todos


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


@tool(
    name="TodoWrite",
    safety="safe",
    description=(
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
        "(pending, in_progress, completed) in every call. Each item is an object: "
        '{"id": str, "content": str, "status": "pending" | "in_progress" | "completed"}.'
    ),
)
def todo_write(todos: list[dict]) -> list[dict]:
    """Creates and manages a structured task list for your current session.

    Each call replaces the entire todo list. Always include all items
    (pending, in_progress, completed) in every call.

    Args:
        todos: The complete updated todo list. Each item should be a dict with keys:
            id (str): Unique identifier for the todo item.
            content (str): Description of the task.
            status (str): One of 'pending', 'in_progress', or 'completed'.
    """
    # Ensure we store a clean copy
    validated = [dict(item) if not isinstance(item, dict) else item for item in todos]
    return [{"text": format_todo_summary(_store_todos(validated))}]
