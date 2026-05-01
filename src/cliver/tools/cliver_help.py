"""CliverHelp tool — lets the LLM look up CLIver command details on demand."""

import importlib
import re
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.command_router import HANDLERS
from cliver.util import get_config_dir

_ALIASES = {"tasks": "task", "skill": "skills"}


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


_COMMAND_SUMMARIES = {
    "agent": "Manage agent instances (list, create, switch, rename, delete)",
    "config": "Manage CLIver configuration (show, validate, set, theme, rate-limit)",
    "cost": "View token usage and cost statistics (session or all-time)",
    "gateway": "Manage the gateway daemon (start, stop, status, platform adapters)",
    "identity": "Manage the agent's identity profile (show, chat, clear)",
    "mcp": "Manage MCP server connections (list, add, set, remove)",
    "model": "Manage LLM model configurations (list, add, set, default, remove)",
    "permissions": "Manage persistent permission rules (rules, mode, add, remove)",
    "provider": "Manage LLM provider endpoints (list, add, set, remove)",
    "session": "Manage conversation sessions (list, load, search, compress, options, permissions)",
    "skills": "Manage agent skills — SKILL.md files (list, show, run, create, update)",
    "task": "Manage and run agent tasks — named prompts with optional cron schedule",
    "workflow": "Manage and execute multi-step workflows (list, show, run, resume, delete)",
}


def _list_commands() -> str:
    """Dynamically list all registered commands with one-line descriptions."""
    lines = ["Available CLIver commands:\n"]
    for name in sorted(HANDLERS.keys()):
        summary = _COMMAND_SUMMARIES.get(name, "(no description)")
        lines.append(f"  /{name:14s} — {summary}")
    lines.append("")
    lines.append("Use CliverHelp(topic='<command>') for full details on any command.")
    lines.append("Use CliverHelp(topic='config_file') for config.yaml reference.")
    return "\n".join(lines)


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


class CliverHelpInput(BaseModel):
    topic: str = Field(
        description=(
            "The topic to look up. Valid values:\n"
            "  'commands'    — List all available CLIver commands with one-line descriptions.\n"
            "  'config_file' — Show config.yaml structure and key sections.\n"
            "  '<command>'   — Show full help for a specific command including all subcommands,\n"
            "                  parameters (with types, required/optional, defaults), and examples.\n"
            "                  Valid command names: agent, config, cost, gateway, identity, mcp,\n"
            "                  model, permissions, provider, session, skills, task, workflow."
        ),
    )


class CliverHelpTool(BaseTool):
    """Look up CLIver command syntax, parameters, and configuration reference."""

    name: str = "CliverHelp"
    description: str = (
        "Look up CLIver's own slash commands and configuration reference. "
        "Returns precise syntax with parameter types, required/optional status, "
        "valid values, defaults, and examples. "
        "Use topic='commands' to list all available commands, "
        "or topic='<command_name>' (e.g. 'task', 'model', 'session') for full "
        "command documentation including all subcommands and parameters. "
        "Use topic='config_file' for config.yaml structure."
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
            return help_text

        available = sorted({"commands", "config_file"} | set(HANDLERS.keys()))
        return f"Unknown topic: '{topic}'. Available: {', '.join(available)}"


cliver_help = CliverHelpTool()
