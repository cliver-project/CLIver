"""
CLI tool execution progress display using Rich.

This module provides a Rich-based renderer for tool execution events.
It is ONLY used by the CLI layer — TaskExecutor and the API layer
have no dependency on this module.
"""

import threading

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from cliver.tool_events import ToolEvent, ToolEventHandler, ToolEventType

# ─── Status Icons ─────────────────────────────────────────────────────────────

_STATUS_ICONS = {
    ToolEventType.TOOL_START: "[bold yellow]⟳[/bold yellow]",
    ToolEventType.TOOL_END: "[bold green]✓[/bold green]",
    ToolEventType.TOOL_ERROR: "[bold red]✗[/bold red]",
}

# ─── Plan Status Icons ────────────────────────────────────────────────────────

_PLAN_ICONS = {
    "pending": "[dim]○[/dim]",
    "in_progress": "[bold yellow]◉[/bold yellow]",
    "completed": "[bold green]●[/bold green]",
    "failed": "[bold red]✗[/bold red]",
}


# ─── Thinking Spinner ────────────────────────────────────────────────────────


class ThinkingIndicator:
    """Manages a Rich spinner shown while the LLM is thinking.

    The spinner is started when inference begins and automatically
    stopped when the first tool call or content chunk arrives.
    """

    def __init__(self, console: Console):
        self._console = console
        self._live: Live | None = None
        self._lock = threading.Lock()

    def start(self, model: str = "") -> None:
        """Start the thinking spinner."""
        with self._lock:
            if self._live is not None:
                return
            label = f"[bold cyan]{model}[/bold cyan] " if model else ""
            spinner = Spinner("dots", text=f"  {label}[dim]Thinking…[/dim]", style="cyan")
            self._live = Live(spinner, console=self._console, transient=True, refresh_per_second=12)
            self._live.start()

    def stop(self) -> None:
        """Stop the thinking spinner."""
        with self._lock:
            if self._live is not None:
                self._live.stop()
                self._live = None

    @property
    def active(self) -> bool:
        return self._live is not None


# ─── Tool Progress Handler ────────────────────────────────────────────────────


def create_tool_progress_handler(
    console: Console,
    thinking: ThinkingIndicator | None = None,
) -> ToolEventHandler:
    """Create a tool event handler that displays progress using Rich.

    Args:
        console: Rich Console instance for output
        thinking: Optional ThinkingIndicator to stop when tools start

    Returns:
        A ToolEventHandler callback for use with TaskExecutor
    """
    # Track whether we're inside a tool execution block for spacing
    state = {"in_block": False}

    def handler(event: ToolEvent) -> None:
        icon = _STATUS_ICONS.get(event.event_type, "")
        tool = f"[cyan]{event.tool_name}[/cyan]"

        if event.event_type == ToolEventType.TOOL_START:
            # Stop thinking spinner when tool execution begins
            if thinking:
                thinking.stop()

            if not state["in_block"]:
                console.print()  # blank line before first tool in a batch
                state["in_block"] = True

            # Tool name and arguments preview
            args_preview = ""
            if event.args:
                parts = []
                for k, v in event.args.items():
                    val = str(v)
                    if len(val) > 60:
                        val = val[:57] + "…"
                    parts.append(f"[dim]{k}=[/dim]{val}")
                args_preview = f"  ({', '.join(parts[:3])})"
                if len(event.args) > 3:
                    args_preview += f" [dim]+{len(event.args) - 3} more[/dim]"

            console.print(f"  {icon} {tool}{args_preview}")

        elif event.event_type == ToolEventType.TOOL_END:
            duration = f"[dim]{event.duration_ms:.0f}ms[/dim]" if event.duration_ms else ""
            console.print(f"  {icon} {tool} {duration}")
            state["in_block"] = False

            # Render plan progress when todo_write completes
            if event.tool_name == "todo_write":
                _render_plan_progress(console)

            console.print()  # blank line after completion

        elif event.event_type == ToolEventType.TOOL_ERROR:
            duration = f"[dim]{event.duration_ms:.0f}ms[/dim]" if event.duration_ms else ""
            console.print(f"  {icon} {tool} {duration}")
            if event.error:
                # Truncate very long errors
                err = event.error if len(event.error) <= 200 else event.error[:197] + "…"
                console.print(f"      [red]{err}[/red]")
            state["in_block"] = False
            console.print()  # blank line after error

    return handler


# ─── Plan Progress Display ────────────────────────────────────────────────────


def _render_plan_progress(console: Console) -> None:
    """Render the current todo plan with clear status markers."""
    from cliver.tools.todo_write import get_current_todos

    todos = get_current_todos()
    if not todos:
        return

    lines = []
    pending = in_progress = completed = failed = 0
    for item in todos:
        status = item.get("status", "pending")
        icon = _PLAN_ICONS.get(status, _PLAN_ICONS["pending"])
        content = item.get("content", "")

        # Style content based on status
        if status == "completed":
            styled = f"[dim strikethrough]{content}[/dim strikethrough]"
        elif status == "in_progress":
            styled = f"[bold]{content}[/bold]"
        elif status == "failed":
            styled = f"[red]{content}[/red]"
        else:
            styled = content

        lines.append(f"  {icon} {styled}")
        if status == "pending":
            pending += 1
        elif status == "in_progress":
            in_progress += 1
        elif status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1

    total = len(todos)
    # Progress bar
    done_ratio = completed / total if total > 0 else 0
    bar_width = 20
    filled = int(done_ratio * bar_width)
    bar = f"[green]{'━' * filled}[/green][dim]{'─' * (bar_width - filled)}[/dim]"

    summary_parts = [f"[green]{completed}[/green]/{total} done"]
    if in_progress:
        summary_parts.append(f"[yellow]{in_progress} active[/yellow]")
    if pending:
        summary_parts.append(f"{pending} pending")
    if failed:
        summary_parts.append(f"[red]{failed} failed[/red]")
    summary = ", ".join(summary_parts)

    console.print()
    console.print(f"  [bold]Plan[/bold]  {bar}  {summary}")
    for line in lines:
        console.print(line)
