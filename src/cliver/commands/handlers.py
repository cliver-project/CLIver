"""
Async TUI command handlers for slash commands.

All handlers follow the signature:
    async def handle_xxx(cliver: "Cliver", args: str) -> None

Handlers output using cliver.output(...) for proper Rich console formatting.
"""

import asyncio
from shlex import split as shell_split
from typing import TYPE_CHECKING

import click

from cliver.commands import list_commands_names

if TYPE_CHECKING:
    from cliver.cli import Cliver


async def handle_help(cliver: "Cliver", args: str) -> None:
    """
    Show help for commands.

    Usage:
        /help          - List all available commands
        /help <cmd>    - Show detailed help for a specific command
    """
    args = args.strip()

    if not args:
        # List all commands
        commands = list_commands_names(cliver._group)
        cliver.output("\n[bold cyan]Available Commands:[/bold cyan]")
        for cmd_name in sorted(commands):
            cmd = cliver._group.commands.get(cmd_name)
            if cmd and not cmd.hidden:
                help_text = cmd.get_short_help_str(limit=60)
                cliver.output(f"  [green]/{cmd_name}[/green] - {help_text}")
        cliver.output("\nUse [yellow]/help <command>[/yellow] for detailed information.")
    else:
        # Show help for specific command
        cmd_name = args.split()[0]
        cmd = cliver._group.commands.get(cmd_name)

        if not cmd:
            cliver.output(f"[red]Error: Unknown command '{cmd_name}'[/red]")
            cliver.output("Use [yellow]/help[/yellow] to see available commands.")
            return

        # Create a minimal context for get_help
        ctx = click.Context(cmd, info_name=cmd_name)
        help_text = cmd.get_help(ctx)
        cliver.output(help_text)


async def handle_model(cliver: "Cliver", args: str) -> None:
    """
    Manage LLM model selection.

    Usage:
        /model              - Show current model
        /model list         - List all available models
        /model <name>       - Set model to <name>
        /model set <name>   - Set model to <name>
    """
    args = args.strip()

    if not args:
        # Show current model
        current = cliver.task_executor.default_model
        cliver.output(f"[cyan]Current model:[/cyan] [green]{current}[/green]")
        return

    parts = args.split()
    command = parts[0]

    if command == "list":
        # List all available models
        models = cliver.config_manager.list_llm_models()
        current = cliver.task_executor.default_model

        cliver.output("\n[bold cyan]Available Models:[/bold cyan]")
        for model_name in sorted(models.keys()):
            marker = "[green]●[/green]" if model_name == current else " "
            cliver.output(f"  {marker} {model_name}")
        cliver.output()
        return

    # Set model (either "/model <name>" or "/model set <name>")
    if command == "set":
        if len(parts) < 2:
            cliver.output("[red]Error: Please specify a model name[/red]")
            cliver.output("Usage: [yellow]/model set <name>[/yellow]")
            return
        model_name = parts[1]
    else:
        model_name = command

    # Validate model exists
    models = cliver.config_manager.list_llm_models()
    if model_name not in models:
        cliver.output(f"[red]Error: Model '{model_name}' not found[/red]")
        cliver.output("Use [yellow]/model list[/yellow] to see available models.")
        return

    # Set the model
    cliver.task_executor.default_model = model_name
    cliver.output(f"[green]✓[/green] Model set to: [cyan]{model_name}[/cyan]")


async def _delegate_to_click(cliver: "Cliver", cmd_name: str, args: str) -> None:
    """
    Migration bridge for complex commands still implemented in Click.

    This delegates execution to the Click command infrastructure, running
    it in a thread executor to avoid blocking the async event loop.
    """
    loop = asyncio.get_event_loop()

    def _run():
        try:
            parts = shell_split(args) if args.strip() else []
        except ValueError:
            # Fallback to simple split if shell_split fails
            parts = args.split() if args.strip() else []

        cli_args = [cmd_name] + parts
        try:
            from cliver.cli import cliver_cli

            cliver_cli(
                args=cli_args,
                prog_name="cliver",
                standalone_mode=False,
                obj=cliver,
            )
        except SystemExit:
            # Click commands often call sys.exit(), ignore it
            pass
        except Exception as e:
            cliver.output(f"[red]Error: {e}[/red]")

    await loop.run_in_executor(None, _run)


# Command handlers that delegate to Click
async def handle_session(cliver: "Cliver", args: str) -> None:
    """Handle /session commands via Click delegation."""
    await _delegate_to_click(cliver, "session", args)


async def handle_config(cliver: "Cliver", args: str) -> None:
    """Handle /config commands via Click delegation."""
    await _delegate_to_click(cliver, "config", args)


async def handle_permissions(cliver: "Cliver", args: str) -> None:
    """Handle /permissions commands via Click delegation."""
    await _delegate_to_click(cliver, "permissions", args)


async def handle_skill(cliver: "Cliver", args: str) -> None:
    """Handle /skill commands via Click delegation."""
    await _delegate_to_click(cliver, "skill", args)


async def handle_identity(cliver: "Cliver", args: str) -> None:
    """Handle /identity commands via Click delegation."""
    await _delegate_to_click(cliver, "identity", args)


async def handle_mcp(cliver: "Cliver", args: str) -> None:
    """Handle /mcp commands via Click delegation."""
    await _delegate_to_click(cliver, "mcp", args)


async def handle_agent(cliver: "Cliver", args: str) -> None:
    """Handle /agent commands via Click delegation."""
    await _delegate_to_click(cliver, "agent", args)


async def handle_cost(cliver: "Cliver", args: str) -> None:
    """Handle /cost commands via Click delegation."""
    await _delegate_to_click(cliver, "cost", args)


async def handle_capabilities(cliver: "Cliver", args: str) -> None:
    """Handle /capabilities commands via Click delegation."""
    await _delegate_to_click(cliver, "capabilities", args)


async def handle_provider(cliver: "Cliver", args: str) -> None:
    """Handle /provider commands via Click delegation."""
    await _delegate_to_click(cliver, "provider", args)


async def handle_task(cliver: "Cliver", args: str) -> None:
    """Handle /task commands via Click delegation."""
    await _delegate_to_click(cliver, "task", args)


async def handle_workflow(cliver: "Cliver", args: str) -> None:
    """Handle /workflow commands via Click delegation."""
    await _delegate_to_click(cliver, "workflow", args)


async def handle_skills(cliver: "Cliver", args: str) -> None:
    """Handle /skills commands via Click delegation."""
    await _delegate_to_click(cliver, "skills", args)
