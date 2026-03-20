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
from rich.panel import Panel
from rich.text import Text

from cliver import __version__

# ─── ASCII Banner ────────────────────────────────────────────────────────────

_BANNER = r"""
   _____ _     _____
  / ____| |   |_   _|
 | |    | |     | |__   _____ _ __
 | |    | |     | |\ \ / / _ \ '__|
 | |____| |_____| |_\ V /  __/ |
  \_____|______|_____\_/ \___|_|
"""

_TIPS = [
    "Type [bold green]/help[/bold green] to see available commands",
    "Use [bold cyan]↑/↓[/bold cyan] to browse command history",
    "Press [bold cyan]Tab[/bold cyan] for command completion",
    "Use [bold green]/model list[/bold green] to see configured models",
    "Use [bold green]/session list[/bold green] to browse past conversations",
    "Use [bold green]/permissions mode[/bold green] to change tool permissions",
    "Use [bold green]/identity chat[/bold green] to set up your agent profile",
    "Use [bold green]/cost[/bold green] to check token usage",
]


def print_banner(console: Console, agent_name: str, default_model: str | None = None) -> None:
    """Print the CLIver ASCII banner with greeting message."""
    # Build banner text
    banner_text = Text()
    for line in _BANNER.strip().splitlines():
        banner_text.append(line + "\n", style="bold cyan")

    # Subtitle line
    subtitle = Text()
    subtitle.append(f"  v{__version__}", style="dim")
    subtitle.append("  •  ", style="dim")
    subtitle.append(agent_name, style="bold white")
    if default_model:
        subtitle.append(f"  •  model: {default_model}", style="dim green")
    banner_text.append(subtitle)

    console.print(Panel(banner_text, border_style="blue", padding=(0, 2)))

    # MOTD or default greeting
    motd = _load_motd()
    if motd:
        console.print(f"  {motd}", style="italic dim")
    else:
        console.print("  Welcome! Your AI-powered CLI assistant is ready.", style="dim")

    # Random tip
    tip = random.choice(_TIPS)
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
