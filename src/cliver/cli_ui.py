"""
CLI UI components for CLIver.

Rich-based UI elements for the interactive terminal experience:
- ASCII banner and greeting
- Spinner/progress indicators
- Status bar helpers
"""

import os
import random
from pathlib import Path

from rich.console import Console

from cliver import __version__

# ─── ASCII Banner ────────────────────────────────────────────────────────────

_BANNER = r"""
  ___ _    ___
 / __| |  |_ _|_ _____ _ _
| (__| |__ | |\ V / -_) '_|
 \___|____|___|\\_/\___|_|
"""

_TIPS = [
    "Type [bold green]/help[/bold green] to see available commands",
    "Use [bold cyan]↑/↓[/bold cyan] to browse command history",
    "Press [bold cyan]Tab[/bold cyan] for command completion",
    "Press [bold cyan]Alt+Enter[/bold cyan] for multi-line input, [bold cyan]Enter[/bold cyan] to submit",
    "Press [bold cyan]Ctrl+G[/bold cyan] to open your editor for longer input",
    "Use [bold green]/model list[/bold green] to see configured models",
    "Use [bold green]/session list[/bold green] to browse past conversations",
    "Use [bold green]/permissions mode[/bold green] to change tool permissions",
    "Use [bold green]/identity chat[/bold green] to set up your agent profile",
    "Use [bold green]/cost[/bold green] to check token usage",
]


def print_banner(console: Console, agent_name: str, default_model: str | None = None) -> None:
    """Print the CLIver ASCII banner with greeting message."""
    for line in _BANNER.strip().splitlines():
        console.print(f"  {line}", style="bold cyan")

    # Subtitle
    parts = [f"\n  [dim]v{__version__}[/dim]  •  [bold white]{agent_name}[/bold white]"]
    if default_model:
        parts.append(f"  •  [dim green]model: {default_model}[/dim green]")
    console.print("".join(parts))
    console.print()

    # MOTD or default greeting
    motd = _load_motd()
    if motd:
        console.print(f"  {motd}", style="italic dim")
    else:
        console.print("  Welcome! Your AI-powered CLI assistant is ready.", style="dim")

    # Random tip
    tip = random.choice(_TIPS)
    console.print()
    console.print(f"  💡 {tip}")
    console.print()


def _load_motd() -> str | None:
    """Load message of the day from /etc/motd or ~/.cliver/motd."""
    for path in [
        Path.home() / ".cliver" / "motd",
        Path(os.environ.get("CLIVER_CONF_DIR", "")) / "motd" if os.environ.get("CLIVER_CONF_DIR") else None,
        Path("/etc/motd"),
    ]:
        if path and path.is_file():
            try:
                content = path.read_text().strip()
                if content:
                    return content.splitlines()[0]  # First line only
            except OSError:
                pass
    return None
