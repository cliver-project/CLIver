"""CliverHelp tool — lets the LLM look up CLIver command details on demand."""

import importlib
import re
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.command_router import HANDLERS
from cliver.util import get_config_dir

_ALIASES = {"tasks": "task", "skills": "skill"}


def _config_file_help() -> str:
    config_dir = get_config_dir()
    return (
        f"CLIver config file: {config_dir}/config.yaml\n\n"
        "Key sections:\n"
        "  models: — LLM model definitions (name, provider, options)\n"
        "  providers: — API endpoints and keys (supports {{ keyring() }} templates)\n"
        "  gateway: — daemon settings (host, port, platforms, session limits)\n"
        "  session: — max_sessions, max_turns_per_session, max_age_days\n"
        "  default_model: — which model to use by default\n"
        "  theme: — UI theme (dark, light, dracula)\n\n"
        "Edit directly with Read/Write tools, or use /config command."
    )


_STATIC_TOPICS = {
    "config_file": _config_file_help,
}


def _list_commands() -> str:
    """Dynamically list all registered commands."""
    names = sorted(HANDLERS.keys())
    return (
        "Available CLIver commands:\n"
        "  " + ", ".join(names) + "\n\n"
        "Use CliverHelp(topic='<command>') for details on any command."
    )


def _get_command_help(command_name: str) -> str:
    """Dynamically load help by calling dispatch(cliver, 'help') with a capture proxy."""
    module_path = HANDLERS.get(command_name)
    if not module_path:
        return f"Unknown command: {command_name}"

    class _OutputCapture:
        def __init__(self):
            self.lines = []

        def output(self, *args, **_kwargs):
            text = str(args[0]) if args else ""
            self.lines.append(re.sub(r"\[/?[a-z][a-z_ ]*\]", "", text))

    try:
        mod = importlib.import_module(module_path)
        capture = _OutputCapture()
        mod.dispatch(capture, "help")
        if capture.lines:
            return f"/{command_name} command:\n" + "\n".join(capture.lines)
        return f"/{command_name} — no help available"
    except Exception as e:
        return f"/{command_name} — error loading help: {e}"


def _filter_task_create_help(help_text: str) -> str:
    """Remove task creation syntax from help output.

    The LLM should use the CreateTask tool instead of shell commands.
    Keep list/run/history/remove but strip the create subcommand line
    and remove 'create' from the usage summary.
    """
    filtered_lines = []
    for line in help_text.splitlines():
        if "create" in line.lower() and ("create <name>" in line.lower() or "--prompt" in line.lower()):
            continue
        line = line.replace("|create|", "|").replace("create|", "").replace("|create", "")
        filtered_lines.append(line)
    filtered_lines.append("\nTo create tasks, use the CreateTask tool (not shell commands).")
    return "\n".join(filtered_lines)


class CliverHelpInput(BaseModel):
    topic: str = Field(
        description="What to look up. Use 'commands' for an overview, "
        "or a specific command name like 'model', 'gateway', 'task'. "
        "Also supports: 'config_file'.",
    )


class CliverHelpTool(BaseTool):
    """Look up CLIver command syntax and configuration reference."""

    name: str = "CliverHelp"
    description: str = (
        "Look up CLIver's own commands and configuration. "
        "Use when you need specific syntax for a CLIver command, "
        "want to know how config.yaml is structured, "
        "or need details about tasks, skills, or the gateway."
    )
    args_schema: Type[BaseModel] = CliverHelpInput
    tags: list = ["admin", "help"]

    def _run(self, topic: str) -> str:
        topic = topic.strip().lower()

        if topic == "commands":
            return _list_commands()

        if topic in _STATIC_TOPICS:
            fn = _STATIC_TOPICS[topic]
            return fn() if callable(fn) else fn

        resolved = _ALIASES.get(topic, topic)
        if resolved in HANDLERS:
            help_text = _get_command_help(resolved)
            if resolved == "task":
                help_text = _filter_task_create_help(help_text)
            return help_text

        available = sorted({"commands", "config_file"} | set(HANDLERS.keys()))
        return f"Unknown topic: '{topic}'. Available: {', '.join(available)}"


cliver_help = CliverHelpTool()
