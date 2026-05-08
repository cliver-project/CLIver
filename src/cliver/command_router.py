"""
CommandRouter -- unified command and LLM query dispatch.

Single dispatch for both TUI and CLI modes:
- command() / command_sync() for slash commands
- query() / query_sync() for LLM prompts
- inject_input() for mid-loop follow-up during LLM queries

Uses ThreadPoolExecutor(max_workers=8) for concurrent task execution.
Slash commands run in parallel. LLM queries are gated to one-at-a-time.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cliver.cli import Cliver

logger = logging.getLogger(__name__)

HANDLERS: dict[str, str] = {
    "clear": "cliver.commands.clear_cmd",
    "model": "cliver.commands.model",
    "config": "cliver.commands.config",
    "mcp": "cliver.commands.mcp",
    "gateway": "cliver.commands.gateway_cmd",
    "session": "cliver.commands.session_cmd",
    "permissions": "cliver.commands.permissions",
    "skills": "cliver.commands.skills",
    "identity": "cliver.commands.identity",
    "profile": "cliver.commands.profile",
    "cost": "cliver.commands.cost",
    "provider": "cliver.commands.provider",
    "task": "cliver.commands.task",
    "workflow": "cliver.commands.workflow_cmd",
}

_task_context = threading.local()


def get_current_task_label() -> str | None:
    """Return the label of the task running on the current thread, or None."""
    return getattr(_task_context, "label", None)


@dataclass
class _TaskEntry:
    future: Future
    label: str
    task_type: str  # "query" or "command"
    task_id: int


class CommandRouter:
    """Unified command and LLM query dispatch with concurrent execution."""

    def __init__(self, cliver: "Cliver"):
        self._cliver = cliver
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cliver-worker")
        self._tasks: dict[int, _TaskEntry] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._pending_input: list[str] = []

    def _register_task(self, future: Future, label: str, task_type: str) -> int:
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            entry = _TaskEntry(future=future, label=label, task_type=task_type, task_id=task_id)
            self._tasks[task_id] = entry

        def _on_done(f, tid=task_id):
            with self._lock:
                self._tasks.pop(tid, None)

        future.add_done_callback(_on_done)
        return task_id

    # -- Sync (CLI mode) ------------------------------------------

    def command_sync(self, name: str, args: str) -> None:
        self._dispatch_command(name, args)

    def query_sync(self, text: str, **kwargs) -> None:
        self._dispatch_query(text, **kwargs)

    # -- Async (TUI mode) -----------------------------------------

    def _dispatch_command_async(self, name: str, args: str) -> int:
        """Submit a command to the executor and return its task ID."""
        future = self._executor.submit(self._dispatch_command, name, args)
        return self._register_task(future, name, "command")

    async def command(self, name: str, args: str) -> None:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            self._dispatch_command,
            name,
            args,
        )
        task_id = self._register_task(future, name, "command")
        try:
            await future
        except Exception as e:
            logger.error(f"Command /{name} failed: {e}")
            self._cliver.output(f"[red]Error: {e}[/red]")
        finally:
            with self._lock:
                self._tasks.pop(task_id, None)

    async def query(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            self._dispatch_query,
            text,
        )
        task_id = self._register_task(future, "chat", "query")
        try:
            await future
        except Exception as e:
            logger.error(f"Query failed: {e}")
            self._cliver.output(f"[red]Error: {e}[/red]")
        finally:
            with self._lock:
                self._tasks.pop(task_id, None)

    # -- Pending input (mid-loop injection) --------------------------

    def inject_input(self, text: str) -> None:
        """Inject text into the running query's Re-Act loop."""
        self._pending_input.append(text)

    def drain_pending(self) -> Optional[str]:
        """Called by the Re-Act loop between iterations to pick up injected input."""
        if self._pending_input:
            return self._pending_input.pop(0)
        return None

    def promote_to_query(self) -> None:
        """Mark the current command as a query session so follow-up input is accepted."""
        with self._lock:
            for entry in self._tasks.values():
                if entry.future.running() and entry.task_type == "command":
                    entry.task_type = "query"
                    break

    # -- State ----------------------------------------------------

    @property
    def has_active_tasks(self) -> bool:
        with self._lock:
            return any(not e.future.done() for e in self._tasks.values())

    @property
    def has_active_query(self) -> bool:
        with self._lock:
            return any(e.task_type == "query" and not e.future.done() for e in self._tasks.values())

    @property
    def is_busy(self) -> bool:
        return self.has_active_tasks

    @property
    def is_query_active(self) -> bool:
        return self.has_active_query

    # -- Cancellation ---------------------------------------------

    def cancel_newest(self) -> bool:
        with self._lock:
            if not self._tasks:
                return False
            newest_id = max(self._tasks.keys())
            entry = self._tasks[newest_id]
        entry.future.cancel()
        self._cliver._cancel_requested = True
        return True

    # -- Internal dispatch ----------------------------------------

    def _dispatch_command(self, name: str, args: str) -> None:
        _task_context.label = name
        try:
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
        finally:
            _task_context.label = None

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
        _task_context.label = "chat"
        try:
            import time as _time

            from langchain_core.messages import AIMessage, HumanMessage

            from cliver.cli_llm_call import LLMCallOptions, llm_call
            from cliver.llm.errors import TaskTimeoutError

            cliver = self._cliver
            model = cliver.session_options.get("model")
            stream = cliver.session_options.get("stream", True)

            start_time = _time.monotonic()
            _real_stdout = None

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
        finally:
            _task_context.label = None

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
