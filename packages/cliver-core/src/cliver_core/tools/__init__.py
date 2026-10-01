"""Builtin tools that ship with cliver-core.

Only the 10 already-pure tools are registered here. CLI-coupled tools
(ask_user_question, cliver_help, create_task, memory, search_sessions,
web_search, image_generate, skill, browse_web, browser_action) are
re-derived in later phases.
"""

from cliver_core.tool import CLIverTool
from cliver_core.tools.docker_run import docker_run
from cliver_core.tools.execute_code import execute_code
from cliver_core.tools.grep_search import grep_search
from cliver_core.tools.list_directory import list_directory
from cliver_core.tools.read_file import read_file
from cliver_core.tools.run_shell_command import run_shell_command
from cliver_core.tools.todo_read import todo_read
from cliver_core.tools.todo_write import todo_write
from cliver_core.tools.web_fetch import web_fetch
from cliver_core.tools.write_file import write_file


def discover_builtin_tools() -> list[CLIverTool]:
    """Return the registered builtin tools.

    Kept as the canonical entry point so the existing registry call sites
    continue to work via this package re-export.
    """
    return [
        read_file,
        write_file,
        list_directory,
        grep_search,
        run_shell_command,
        execute_code,
        docker_run,
        todo_read,
        todo_write,
        web_fetch,
    ]


__all__ = [
    "read_file",
    "write_file",
    "list_directory",
    "grep_search",
    "run_shell_command",
    "execute_code",
    "docker_run",
    "todo_read",
    "todo_write",
    "web_fetch",
    "discover_builtin_tools",
]
