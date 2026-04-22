"""CliverHelp tool — lets the LLM look up CLIver command details on demand."""

import importlib
import re
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_COMMAND_MODULES = {
    "model": "cliver.commands.model",
    "config": "cliver.commands.config",
    "mcp": "cliver.commands.mcp",
    "gateway": "cliver.commands.gateway_cmd",
    "session": "cliver.commands.session_cmd",
    "permissions": "cliver.commands.permissions",
    "skill": "cliver.commands.skill_cmd",
    "skills": "cliver.commands.skills",
    "identity": "cliver.commands.identity",
    "agent": "cliver.commands.agent",
    "cost": "cliver.commands.cost",
    "provider": "cliver.commands.provider",
    "task": "cliver.commands.task",
    "workflow": "cliver.commands.workflow_cmd",
}

# Static topics for non-command reference
_STATIC_TOPICS = {
    "commands": (
        "Available CLIver commands:\n"
        "  model, config, gateway, session, permissions, mcp,\n"
        "  skill, skills, identity, agent, cost, provider, task, workflow\n\n"
        "Use CliverHelp(topic='<command>') for details on any command."
    ),
    "config_file": (
        "CLIver config file: ~/.cliver/config.yaml\n\n"
        "Key sections:\n"
        "  models: — LLM model definitions (name, provider, options)\n"
        "  providers: — API endpoints and keys (supports {{ keyring() }} templates)\n"
        "  gateway: — daemon settings (host, port, platforms, session limits)\n"
        "  session: — max_sessions, max_turns_per_session, max_age_days\n"
        "  default_model: — which model to use by default\n"
        "  theme: — UI theme (dark, light, dracula)\n\n"
        "Edit directly with Read/Write tools, or use /config command."
    ),
    "tasks": (
        "Tasks are scheduled or one-shot jobs run by the gateway daemon.\n\n"
        "Task YAML files live in: ~/.cliver/agents/<agent>/tasks/\n"
        "Each file defines: name, prompt, schedule (cron), model (optional)\n\n"
        "Example task YAML:\n"
        "  name: daily-summary\n"
        "  prompt: Summarize today's news about AI\n"
        "  schedule: '0 18 * * *'\n"
        "  model: qwen\n\n"
        "Commands: /task list, /task create, /task run, /task remove"
    ),
}


def _get_command_help(command_name: str) -> str:
    """Dynamically load help by calling dispatch(cliver, 'help') with a capture proxy."""
    module_path = _COMMAND_MODULES.get(command_name)
    if not module_path:
        return f"Unknown command: {command_name}"

    class _OutputCapture:
        """Minimal Cliver proxy that captures output() calls."""

        def __init__(self):
            self.lines = []

        def output(self, *args, **kwargs):
            text = str(args[0]) if args else ""
            self.lines.append(re.sub(r"\[/?[^\]]*\]", "", text))

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
        description="What to look up. Use 'commands' for an overview, "
        "or a specific command name like 'model', 'gateway', 'task'. "
        "Also supports: 'config_file', 'skills', 'tasks'.",
    )


class CliverHelpTool(BaseTool):
    """Look up CLIver command syntax and configuration reference."""

    name: str = "CliverHelp"
    description: str = (
        "Look up CLIver's own commands and configuration. "
        "Use when you need specific syntax for a CLIver command, "
        "want to know how config.yaml is structured, "
        "or need details about skills, tasks, or the gateway.\n\n"
        "Topics: commands, config_file, skills, tasks, gateway, "
        "model, config, session, permissions, mcp, provider, "
        "agent, identity, cost, skill, workflow"
    )
    args_schema: Type[BaseModel] = CliverHelpInput
    tags: list = ["admin", "help"]

    def _run(self, topic: str) -> str:
        topic = topic.strip().lower()

        if topic in _STATIC_TOPICS:
            return _STATIC_TOPICS[topic]

        if topic in _COMMAND_MODULES:
            return _get_command_help(topic)

        available = sorted(set(list(_STATIC_TOPICS.keys()) + list(_COMMAND_MODULES.keys())))
        return f"Unknown topic: '{topic}'. Available: {', '.join(available)}"


cliver_help = CliverHelpTool()
