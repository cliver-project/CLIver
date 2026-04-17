"""
CLI tool execution progress display using Rich.

This module provides a Rich-based renderer for tool execution events.
It is ONLY used by the CLI layer — AgentCore and the API layer
have no dependency on this module.
"""

import threading

from rich.console import Console

from cliver.tool_events import ToolEvent, ToolEventHandler, ToolEventType

# ─── Status Icons ─────────────────────────────────────────────────────────────

_STATUS_ICONS = {
    ToolEventType.TOOL_START: "[bold yellow]⟳[/bold yellow]",
    ToolEventType.TOOL_END: "[bold green]✓[/bold green]",
    ToolEventType.TOOL_ERROR: "[bold red]✗[/bold red]",
    ToolEventType.MODEL_RETRY: "[bold yellow]↻[/bold yellow]",
    ToolEventType.MODEL_COMPRESS: "[bold blue]⊘[/bold blue]",
    ToolEventType.MODEL_FALLBACK: "[bold magenta]⇢[/bold magenta]",
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
    """Shows a live animated thinking indicator while the LLM is processing.

    Started when inference begins, stopped on first tool call or content chunk.
    Runs a background thread that prints color-cycling dots.
    """

    _COLORS = ["#ff6b6b", "#ffa06b", "#ffd86b", "#6bffa0", "#6bd8ff", "#a06bff"]
    _PHRASES = [
        "Thinking…",
        "Reasoning…",
        "Analyzing…",
        "Considering…",
        "Processing…",
        "Reflecting…",
        "Pondering…",
        "Evaluating…",
        "Working…",
        "Figuring out…",
    ]

    def __init__(self, console: Console):
        self._console = console
        self._active = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._model = ""

    def start(self, model: str = "") -> None:
        """Start the animated thinking indicator."""
        with self._lock:
            if self._active:
                return
            self._active = True
            self._model = model
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def _animate(self) -> None:
        """Background animation loop."""
        import random
        import sys
        import time

        label = f"{self._model} " if self._model else ""
        frame = 0
        phrase = random.choice(self._PHRASES)
        while self._active:
            color = self._COLORS[frame % len(self._COLORS)]
            dots = "●" * ((frame % 3) + 1)
            # Switch phrase every ~3 seconds (10 frames × 0.3s)
            if frame > 0 and frame % 10 == 0:
                phrase = random.choice(self._PHRASES)
            sys.stdout.write(f"\r  \033[38;2;{_hex_to_rgb(color)}m{dots}\033[0m \033[2m{label}{phrase}\033[0m  ")
            sys.stdout.flush()
            frame += 1
            time.sleep(0.3)
        # Clear the line when done
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    def stop(self, blank_line: bool = True) -> None:
        """Stop the animated indicator.

        Args:
            blank_line: Print an extra blank line after stopping (before LLM content).
                        Set to False when the spinner will be restarted immediately.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        if blank_line:
            print()

    @property
    def active(self) -> bool:
        return self._active


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#rrggbb' to 'r;g;b' for ANSI escape."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}"


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
        A ToolEventHandler callback for use with AgentCore
    """
    # Track whether we're inside a tool execution block for spacing
    state = {"in_block": False}

    def handler(event: ToolEvent) -> None:
        icon = _STATUS_ICONS.get(event.event_type, "")
        tool = f"[cyan]{event.tool_name}[/cyan]"

        if event.event_type == ToolEventType.TOOL_START:
            # Stop thinking spinner when tool execution begins (no blank line — spinner restarts after)
            if thinking:
                thinking.stop(blank_line=False)

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

            # Show tool result preview
            if event.result and event.result not in ("denied", "(no output)"):
                _render_tool_result(console, event.result)

            state["in_block"] = False

            # Render plan progress when TodoWrite completes
            if event.tool_name == "TodoWrite":
                _render_plan_progress(console)

            console.print()  # blank line after completion

            # Restart spinner while LLM processes the tool result
            if thinking:
                thinking.start(thinking._model)

        elif event.event_type == ToolEventType.TOOL_ERROR:
            duration = f"[dim]{event.duration_ms:.0f}ms[/dim]" if event.duration_ms else ""
            console.print(f"  {icon} {tool} {duration}")
            if event.error:
                # Truncate very long errors
                err = event.error if len(event.error) <= 200 else event.error[:197] + "…"
                console.print(f"      [red]{err}[/red]")
            state["in_block"] = False
            console.print()  # blank line after error

            # Restart spinner while LLM processes the error
            if thinking:
                thinking.start(thinking._model)

        elif event.event_type == ToolEventType.MODEL_FALLBACK:
            if thinking:
                thinking.stop(blank_line=False)
            model = f"[bold magenta]{event.tool_name}[/bold magenta]"
            reason = f"[dim]{event.result}[/dim]" if event.result else ""
            console.print(f"\n  {icon} Falling back to {model}  {reason}\n")
            if thinking:
                thinking.start(event.tool_name)

        elif event.event_type == ToolEventType.MODEL_RETRY:
            if thinking:
                thinking.stop(blank_line=False)
            model = f"[yellow]{event.tool_name}[/yellow]"
            reason = f"[dim]{event.result}[/dim]" if event.result else ""
            console.print(f"\n  {icon} Retrying {model}  {reason}\n")
            if thinking:
                thinking.start(event.tool_name)

        elif event.event_type == ToolEventType.MODEL_COMPRESS:
            if thinking:
                thinking.stop(blank_line=False)
            model = f"[blue]{event.tool_name}[/blue]"
            console.print(f"\n  {icon} Compressing context for {model}\n")
            if thinking:
                thinking.start(event.tool_name)

    return handler


# ─── Tool Result Display ─────────────────────────────────────────────────────

_MAX_RESULT_LINES = 15
_MAX_LINE_WIDTH = 120


def _render_tool_result(console: Console, result: str) -> None:
    """Render a compact preview of the tool result output."""
    lines = result.splitlines()
    total = len(lines)

    # Truncate lines that are too wide
    display_lines = []
    for line in lines[:_MAX_RESULT_LINES]:
        if len(line) > _MAX_LINE_WIDTH:
            line = line[:_MAX_LINE_WIDTH] + "…"
        display_lines.append(line)

    body = "\n".join(display_lines)
    if total > _MAX_RESULT_LINES:
        body += f"\n[dim]… ({total - _MAX_RESULT_LINES} more lines)[/dim]"

    console.print(f"      [dim]{body}[/dim]")


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
