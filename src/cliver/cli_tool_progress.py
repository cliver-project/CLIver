"""
CLI tool execution progress display using Rich.

This module provides a Rich-based renderer for tool execution events.
It is ONLY used by the CLI layer — TaskExecutor and the API layer
have no dependency on this module.
"""

from rich.console import Console

from cliver.tool_events import ToolEvent, ToolEventHandler, ToolEventType

# Status indicators (similar to qwen-code)
_STATUS_ICONS = {
    ToolEventType.TOOL_START: "[bold yellow]⟳[/bold yellow]",
    ToolEventType.TOOL_END: "[bold green]✓[/bold green]",
    ToolEventType.TOOL_ERROR: "[bold red]✗[/bold red]",
}


def create_tool_progress_handler(console: Console) -> ToolEventHandler:
    """Create a tool event handler that displays progress using Rich.

    Args:
        console: Rich Console instance for output

    Returns:
        A ToolEventHandler callback for use with TaskExecutor
    """
    # Track whether we're inside a tool execution block for spacing
    state = {"in_block": False}

    def handler(event: ToolEvent) -> None:
        icon = _STATUS_ICONS.get(event.event_type, "")
        tool = f"[cyan]{event.tool_name}[/cyan]"

        if event.event_type == ToolEventType.TOOL_START:
            if not state["in_block"]:
                console.print()  # blank line before first tool in a batch
                state["in_block"] = True
            args_preview = ""
            if event.args:
                parts = []
                for k, v in event.args.items():
                    val = str(v)
                    if len(val) > 40:
                        val = val[:37] + "..."
                    parts.append(f"{k}={val}")
                args_preview = f" [dim]({', '.join(parts[:3])})[/dim]"
            console.print(f"  {icon} {tool}{args_preview}")

        elif event.event_type == ToolEventType.TOOL_END:
            duration = f"[dim]{event.duration_ms:.0f}ms[/dim]" if event.duration_ms else ""
            console.print(f"  {icon} {tool} {duration}")
            state["in_block"] = False
            console.print()  # blank line after completion

        elif event.event_type == ToolEventType.TOOL_ERROR:
            duration = f"[dim]{event.duration_ms:.0f}ms[/dim]" if event.duration_ms else ""
            error_msg = f"[red]{event.error}[/red]" if event.error else ""
            console.print(f"  {icon} {tool} {duration}")
            if error_msg:
                console.print(f"      {error_msg}")
            state["in_block"] = False
            console.print()  # blank line after error

    return handler
