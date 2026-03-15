"""Built-in todo_read tool for reading current plan progress."""

from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from cliver.tools.todo_write import format_todo_summary, get_current_todos


class TodoReadInput(BaseModel):
    """Input schema for the todo_read tool (no parameters needed)."""

    pass


class TodoReadTool(BaseTool):
    """Read the current plan/todo list without modifying it."""

    name: str = "todo_read"
    description: str = (
        "Read the current plan/todo list to check progress. "
        "Returns the full todo list with status of each item. "
        "Use this to review your plan before deciding what to do next, "
        "especially after completing a step or when resuming work."
    )
    args_schema: Type[BaseModel] = TodoReadInput
    tags: list = ["think", "planning", "task"]

    def _run(self) -> str:
        return format_todo_summary(get_current_todos())


todo_read = TodoReadTool()
