"""Future-style handles for long-running media generation jobs.

``AgentCore.submit_*()`` methods start generation in the background and
return a :class:`MediaJob` immediately, so callers can keep working and
collect the result later::

    job = agent.submit_image("a red panda on a skateboard")
    ...                       # do other work
    response = await job.result(timeout=120)
"""

from __future__ import annotations

import asyncio
import time
from typing import Generic, TypeVar
from uuid import uuid4

T = TypeVar("T")


class MediaJob(Generic[T]):
    """A handle to an in-flight media job running on the event loop.

    - ``status`` — ``"running"`` / ``"done"`` / ``"failed"`` / ``"cancelled"``
    - ``await job.result()`` — wait for (and return) the result
    - ``job.result(timeout=...)`` — bounded wait; the job keeps running
    - ``job.cancel()`` — abort the job
    """

    def __init__(self, purpose: str, task: "asyncio.Task[T]") -> None:
        self.id = uuid4().hex
        self.purpose = purpose
        self.created_at = time.time()
        self._task = task

    @property
    def status(self) -> str:
        if self._task.cancelled():
            return "cancelled"
        if self._task.done():
            return "failed" if self._task.exception() is not None else "done"
        return "running"

    def done(self) -> bool:
        """True when the job has finished (successfully or not)."""
        return self._task.done()

    def cancel(self) -> bool:
        """Request cancellation. Returns True if the job was cancelled."""
        return self._task.cancel()

    async def result(self, timeout: float | None = None) -> T:
        """Await the job's result.

        With a *timeout*, raises ``asyncio.TimeoutError`` when it expires —
        the job itself keeps running (it is shielded from the cancellation).
        """
        if timeout is None:
            return await self._task
        return await asyncio.wait_for(asyncio.shield(self._task), timeout)

    def __repr__(self) -> str:
        return f"MediaJob(id={self.id!r}, purpose={self.purpose!r}, status={self.status!r})"
