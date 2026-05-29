"""Help command — show available slash commands and their descriptions."""

from cliver.command_router import HANDLERS


def dispatch(cliver, args: str):
    """Show available slash commands and their descriptions."""
    commands = sorted(HANDLERS.keys())
    cliver.output("Available slash commands:")
    for cmd in commands:
        cliver.output(f"  /{cmd}")
    cliver.output("")
    cliver.output("Type /<command> --help for detailed usage.")
    cliver.output("For tool help, ask the agent: 'What tools are available?'")
