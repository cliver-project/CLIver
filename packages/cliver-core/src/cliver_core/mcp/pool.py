"""Exclusive-lease session pool for streamable-http MCP sessions.

The high-level ``mcp.Client`` enters and exits anyio cancel scopes, which
must happen in the *same task* that entered them.  A pool therefore runs one
manager task that owns every session open and close; callers only ever lease
a session the manager opened, and hand it back through :meth:`release`.

``size`` is the maximum number of concurrent tool calls for one server.  A
call beyond the cap waits for an idle session — normally for another call to
finish.  A session that raised (transport failure) is closed and replaced; an
MCP ``is_error`` result is tool-level, not transport-level, and keeps the
session.

The pool is loop-bound: acquire/release/close on another loop raise
``RuntimeError``.  ``close()`` closes every session — leased ones included,
which is how the SDK cancels in-flight calls — so drain them first when their
errors must not be observed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from contextlib import AsyncExitStack
from typing import Any, AsyncContextManager, Callable

logger = logging.getLogger(__name__)

__all__ = ["SessionPool"]


class SessionPool:
    """A loop-bound pool of open sessions with exclusive leases.

    ``open_session`` is a zero-argument callable returning an async context
    manager that yields one connected session (for MCP:
    ``adapter._connected``).  ``size`` is the maximum number of concurrent
    leases and must be a positive integer.
    """

    def __init__(self, open_session: Callable[[], AsyncContextManager[Any]], size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("session pool size must be a positive integer")
        self._open_session = open_session
        self._size = size
        self._loop = asyncio.get_running_loop()
        self._condition = asyncio.Condition()
        self._idle: deque[Any] = deque()
        self._leased = 0
        self._closed = False
        self._close_task: asyncio.Task | None = None
        self._commands: asyncio.Queue[tuple[str, Any, asyncio.Future | None]] = asyncio.Queue()
        # Sessions the manager opened -> their exit stacks.  Idle and leased
        # sessions alike stay here until evicted or closed.
        self._sessions: dict[int, tuple[Any, AsyncExitStack]] = {}
        self._manager: asyncio.Task | None = None

    # ── Leases ─────────────────────────────────────────────

    async def acquire(self) -> Any:
        """Lease one session — an idle one, or a newly opened one.

        Waits while ``size`` sessions are leased.  Raises ``RuntimeError``
        when the pool is (or becomes) closed; an open failure propagates to
        the caller and frees the slot for the next acquire.
        """
        self._check_loop()
        async with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("MCP session pool is closed")
                if self._idle:
                    session = self._idle.popleft()
                    self._leased += 1
                    return session
                if self._leased < self._size:
                    self._leased += 1
                    break
                await self._condition.wait()
        try:
            return await self._request("open")
        except BaseException:
            async with self._condition:
                self._leased -= 1
                self._condition.notify()
            raise

    async def release(self, session: Any, *, ok: bool = True) -> None:
        """Return a lease.

        ``ok=True`` returns the session to the idle queue for reuse.  A
        transport failure releases with ``ok=False``: the manager closes and
        forgets the session, and the next acquire opens a replacement.
        """
        self._check_loop()
        if self._closed:
            return  # close() already closed every session; nothing to return
        if ok:
            async with self._condition:
                self._idle.append(session)
                self._leased -= 1
                self._condition.notify()
            return
        try:
            await self._request("close", session)
        finally:
            async with self._condition:
                self._leased -= 1
                self._condition.notify()

    # ── Lifecycle ──────────────────────────────────────────

    async def close(self) -> None:
        """Close every session and stop the manager — idempotent, single-flight."""
        self._check_loop()
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()  # waiting acquirers wake with RuntimeError
        manager = self._manager
        if manager is not None and not manager.done():
            # Bypass _request(): it refuses commands once closed, but stop is
            # the one command close() must deliver.  Awaiting the manager
            # covers the stop future.
            self._commands.put_nowait(("stop", None, None))
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(manager)

    # ── Manager task ───────────────────────────────────────

    def _check_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("SessionPool is bound to another event loop; create one pool per loop")

    def _ensure_manager(self) -> None:
        if self._closed:
            raise RuntimeError("MCP session pool is closed")
        if self._manager is None or self._manager.done():
            self._manager = self._loop.create_task(self._run_manager())

    async def _request(self, command: str, payload: Any = None) -> Any:
        self._ensure_manager()
        future: asyncio.Future = self._loop.create_future()
        await self._commands.put((command, payload, future))
        return await future

    async def _run_manager(self) -> None:
        """Own every open and close — the SDK's cancel scopes need task affinity."""
        while True:
            command, payload, future = await self._commands.get()
            if command == "stop":
                for _, stack in list(self._sessions.values()):
                    with contextlib.suppress(Exception):
                        await stack.aclose()
                self._sessions.clear()
                if future is not None and not future.done():
                    future.set_result(None)
                return
            if command == "open":
                session: Any = None
                stack = AsyncExitStack()
                try:
                    session = await stack.enter_async_context(self._open_session())
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        await stack.aclose()
                    if future is not None and not future.done():
                        future.cancel()
                    raise
                except Exception as exc:
                    with contextlib.suppress(Exception):
                        await stack.aclose()
                    if future is not None and not future.done():
                        future.set_exception(exc)
                    continue
                self._sessions[id(session)] = (session, stack)
                if future is None or future.done():
                    # The acquirer went away (cancelled or timed out) — recycle.
                    entry = self._sessions.pop(id(session), None)
                    if entry is not None:
                        with contextlib.suppress(Exception):
                            await entry[1].aclose()
                    continue
                future.set_result(session)
                continue
            if command == "close":
                entry = self._sessions.pop(id(payload), None)
                with contextlib.suppress(Exception):
                    if entry is not None:
                        await entry[1].aclose()
                if future is not None and not future.done():
                    future.set_result(None)
