"""
CommandRouter -- unified command and LLM query dispatch.

Single dispatch for both TUI and CLI modes:
- command() / command_sync() for slash commands
- query() / query_sync() for LLM prompts
- inject_input() for follow-up during LLM queries

Uses ThreadPoolExecutor(max_workers=1) for the worker thread.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cliver.cli import Cliver

logger = logging.getLogger(__name__)

HANDLERS: dict[str, str] = {
    "model": "cliver.commands.model",
    "config": "cliver.commands.config",
    "mcp": "cliver.commands.mcp",
    "gateway": "cliver.commands.gateway_cmd",
    "session": "cliver.commands.session_cmd",
    "permissions": "cliver.commands.permissions",
    "skills": "cliver.commands.skills",
    "identity": "cliver.commands.identity",
    "agent": "cliver.commands.agent",
    "cost": "cliver.commands.cost",
    "provider": "cliver.commands.provider",
    "task": "cliver.commands.task",
    "workflow": "cliver.commands.workflow_cmd",
}


class CommandRouter:
    """Unified command and LLM query dispatch."""

    def __init__(self, cliver: "Cliver"):
        self._cliver = cliver
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cliver-worker")
        self._active_future: Optional[Future] = None
        self._is_query = False
        self._pending_input: list[str] = []

    # -- Sync (CLI mode) ------------------------------------------

    def command_sync(self, name: str, args: str) -> None:
        self._dispatch_command(name, args)

    def query_sync(self, text: str, **kwargs) -> None:
        self._dispatch_query(text, **kwargs)

    # -- Async (TUI mode) -----------------------------------------

    async def command(self, name: str, args: str) -> None:
        loop = asyncio.get_running_loop()
        self._is_query = False
        self._active_future = loop.run_in_executor(
            self._executor,
            self._dispatch_command,
            name,
            args,
        )
        try:
            await self._active_future
        except Exception as e:
            logger.error(f"Command /{name} failed: {e}")
            self._cliver.output(f"[red]Error: {e}[/red]")
        finally:
            self._active_future = None

    async def query(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        self._is_query = True
        self._active_future = loop.run_in_executor(
            self._executor,
            self._dispatch_query,
            text,
        )
        try:
            await self._active_future
        except Exception as e:
            logger.error(f"Query failed: {e}")
            self._cliver.output(f"[red]Error: {e}[/red]")
        finally:
            self._active_future = None
            self._is_query = False

    # -- Pending input --------------------------------------------

    def inject_input(self, text: str) -> None:
        self._pending_input.append(text)

    def drain_pending(self) -> Optional[str]:
        if self._pending_input:
            return self._pending_input.pop(0)
        return None

    # -- State ----------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._active_future is not None and not self._active_future.done()

    @property
    def is_query_active(self) -> bool:
        return self.is_busy and self._is_query

    # -- Internal dispatch ----------------------------------------

    def _dispatch_command(self, name: str, args: str) -> None:
        module_path = HANDLERS.get(name)
        if not module_path:
            self._cliver.output(f"[yellow]Unknown command: /{name}[/yellow]")
            return
        try:
            mod = importlib.import_module(module_path)
            mod.dispatch(self._cliver, args)
        except Exception as e:
            logger.exception(f"Command /{name} error")
            self._cliver.output(f"[red]Error in /{name}: {e}[/red]")

    def _dispatch_query(
        self,
        text: str,
        images: list[str] | None = None,
        audio_files: list[str] | None = None,
        video_files: list[str] | None = None,
        files: list[str] | None = None,
        output_format: str | None = None,
        timeout_s: int | None = None,
    ) -> None:
        import time as _time

        from langchain_core.messages import AIMessage, HumanMessage

        from cliver.cli_llm_call import LLMCallOptions, llm_call
        from cliver.llm.errors import TaskTimeoutError

        cliver = self._cliver
        model = cliver.session_options.get("model")
        stream = cliver.session_options.get("stream", True)

        start_time = _time.monotonic()
        _real_stdout = None

        # Setup JSON output mode if needed
        if output_format == "json":
            import io
            import sys

            from rich.console import Console

            _real_stdout = sys.stdout
            cliver.console = Console(file=io.StringIO(), quiet=True)
            if hasattr(cliver, "thinking") and cliver.thinking:
                cliver.thinking = None

        cliver.conversation_messages.append(HumanMessage(content=text))
        cliver.record_turn("user", text)

        def on_response(response_text: str) -> None:
            cliver.conversation_messages.append(AIMessage(content=response_text))
            cliver.record_turn("assistant", response_text)

        opts = LLMCallOptions(
            user_input=text,
            model=model,
            stream=stream,
            images=images or [],
            audio_files=audio_files or [],
            video_files=video_files or [],
            files=files or [],
            conversation_history=cliver.conversation_messages[:-1] or None,
            on_response=on_response,
            on_pending_input=self.drain_pending,
            auto_fallback=True,
            timeout_s=timeout_s,
        )

        try:
            result = llm_call(cliver, opts)
        except TaskTimeoutError as e:
            if output_format == "json":
                import sys as _sys

                duration = _time.monotonic() - start_time
                self._emit_json(
                    _real_stdout or _sys.stdout,
                    False,
                    e.partial_result or "",
                    model,
                    cliver.token_tracker,
                    duration,
                    error=str(e),
                    timeout=True,
                )
                return
            cliver.output(f"[yellow]Timeout: {e}[/yellow]")
            if e.partial_result:
                cliver.output(f"\nPartial result:\n{e.partial_result}")
            return
        except Exception as e:
            if output_format == "json":
                import sys as _sys

                duration = _time.monotonic() - start_time
                self._emit_json(
                    _real_stdout or _sys.stdout,
                    False,
                    "",
                    model,
                    cliver.token_tracker,
                    duration,
                    error=str(e),
                )
                return
            logger.exception("LLM query failed")
            cliver.output(f"[red]Error: {e}[/red]")
            return

        if output_format == "json":
            import sys as _sys

            duration = _time.monotonic() - start_time
            self._emit_json(
                _real_stdout or _sys.stdout,
                result.success,
                result.text or "",
                model,
                cliver.token_tracker,
                duration,
                error=result.error,
            )

    @staticmethod
    def _emit_json(stdout, success, output, model, token_tracker, duration, error=None, timeout=False):
        """Emit a single JSON result object to stdout."""
        import json

        tokens = None
        if token_tracker and hasattr(token_tracker, "last_usage") and token_tracker.last_usage:
            last = token_tracker.last_usage
            tokens = {
                "input": last.input_tokens,
                "output": last.output_tokens,
                "total": last.total_tokens,
            }

        data = {
            "success": success,
            "output": output,
            "error": error or "",
            "model": model,
            "tokens": tokens,
            "duration_s": round(duration, 2),
        }
        if timeout:
            data["timeout"] = True

        stdout.write(json.dumps(data) + "\n")
        stdout.flush()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
