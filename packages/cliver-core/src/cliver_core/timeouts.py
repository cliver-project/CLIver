"""Cancel-based deadlines for Python 3.10+.

``asyncio.wait_for`` cannot be used where the awaited code enters anyio
cancel scopes (the MCP client does): on Python < 3.12 it runs the awaited
coroutine in a *new* task, so scopes are entered in one task and exited in
another (``RuntimeError: Attempted to exit cancel scope in a different
task``).  Cancelling the current task keeps every context manager unwinding
in the task that entered it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

__all__ = ["timeout_after"]


@asynccontextmanager
async def timeout_after(
    seconds: float,
    *,
    message: str | None = None,
    error_factory: Callable[[], BaseException] | None = None,
) -> AsyncIterator[None]:
    """Cancel the current task after ``seconds``; raise on expiry.

    The raised exception is ``error_factory()`` when given (a fresh instance
    per expiry, used by the run deadline for ``RunTimeoutError``), otherwise
    ``TimeoutError(message or f"timed out after {seconds:g}s")``.  Caller
    cancellation is never converted: a ``CancelledError`` that arrives while
    the timer has not fired propagates unchanged.
    """
    task = asyncio.current_task()
    timed_out = False

    def _cancel() -> None:
        nonlocal timed_out
        timed_out = True
        if task is not None:
            task.cancel()

    handle = asyncio.get_running_loop().call_later(seconds, _cancel)
    try:
        yield
    except asyncio.CancelledError:
        if not timed_out:
            raise
        uncancel = getattr(task, "uncancel", None)
        if uncancel is not None:  # Python 3.11+
            uncancel()
        if error_factory is not None:
            raise error_factory() from None
        raise TimeoutError(message or f"timed out after {seconds:g}s") from None
    finally:
        handle.cancel()
