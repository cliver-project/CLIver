"""
CommandDispatcher — async command routing for the TUI.
Replaces Click-based routing in the interactive TUI loop.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Coroutine, Dict, Optional

if TYPE_CHECKING:
    from cliver.cli import Cliver

logger = logging.getLogger(__name__)

AsyncHandler = Callable[["Cliver", str], Coroutine]


class CommandDispatcher:
    def __init__(self, cliver: "Cliver"):
        self._cliver = cliver
        self._handlers: Dict[str, AsyncHandler] = {}
        self._active_chat_task: Optional[asyncio.Task] = None
        self._pending_input: list[str] = []
        self._chat_runner: Optional[Callable[[str], Coroutine]] = None

    def register(self, name: str, handler: AsyncHandler) -> None:
        """Register a command handler. Name should not include leading /."""
        self._handlers[name.lower()] = handler

    def set_chat_runner(self, runner: Callable[[str], Coroutine]) -> None:
        """Set the async function that runs a chat turn."""
        self._chat_runner = runner

    async def dispatch(self, line: str) -> Optional[str]:
        """Route input. Returns "exit" if user wants to quit, None otherwise."""
        stripped = line.strip()
        if not stripped:
            return None

        if stripped.lower() in ("exit", "quit", "/exit", "/quit"):
            if self._active_chat_task and not self._active_chat_task.done():
                self._active_chat_task.cancel()
            return "exit"

        if stripped.startswith("/"):
            cmd_text = stripped[1:]
            if not cmd_text:
                return None
            parts = cmd_text.split(None, 1)
            cmd_name = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""

            handler = self._handlers.get(cmd_name)
            if handler:
                asyncio.create_task(handler(self._cliver, cmd_args))
            else:
                self._cliver.output(f"[yellow]Unknown command: /{cmd_name}[/yellow]")
            return None

        # Plain text → chat
        self._cliver.output(f"\n[bold green]❯[/bold green] [bold]{stripped}[/bold]")
        await self._handle_chat_input(stripped)
        return None

    async def _handle_chat_input(self, text: str) -> None:
        if self._active_chat_task and not self._active_chat_task.done():
            self._pending_input.append(text)
            logger.info("Appended pending input (%d chars), %d pending total", len(text), len(self._pending_input))
            return

        if self._chat_runner is None:
            self._cliver.output("[red]Chat not available.[/red]")
            return

        self._active_chat_task = asyncio.create_task(self._run_chat_loop(text))

    async def _run_chat_loop(self, initial_text: str) -> None:
        try:
            await self._chat_runner(initial_text)
            while self._pending_input:
                next_text = self._pending_input.pop(0)
                self._cliver.output(f"\n[bold green]❯[/bold green] [bold]{next_text}[/bold]")
                await self._chat_runner(next_text)
        except asyncio.CancelledError:
            logger.info("Chat task cancelled")
        except Exception as e:
            self._cliver.output(f"[red]Error: {e}[/red]")
            logger.exception("Chat task failed")
        finally:
            self._active_chat_task = None

    def drain_pending(self) -> Optional[str]:
        """Return next pending input, or None. Passed as on_pending_input callback."""
        if self._pending_input:
            return self._pending_input.pop(0)
        return None

    @property
    def is_chat_active(self) -> bool:
        return self._active_chat_task is not None and not self._active_chat_task.done()
