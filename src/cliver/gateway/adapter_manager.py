"""Adapter lifecycle manager with exponential backoff reconnect."""

import asyncio
import logging
from typing import Any, Callable, Coroutine, List

from cliver.gateway.platform_adapter import PlatformAdapter

logger = logging.getLogger(__name__)

MessageCallback = Callable[..., Coroutine[Any, Any, None]]


class AdapterManager:
    """Manages platform adapter connections with resilient lifecycle."""

    def __init__(
        self,
        adapters: List[PlatformAdapter],
        on_message: MessageCallback,
        initial_backoff: float = 5.0,
        max_backoff: float = 300.0,
        start_timeout: float = 30.0,
        stop_timeout: float = 10.0,
    ):
        self._adapters = adapters
        self._on_message = on_message
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._start_timeout = start_timeout
        self._stop_timeout = stop_timeout
        self._tasks: List[asyncio.Task] = []
        self._connected: set[str] = set()

    @property
    def connected_platforms(self) -> list[str]:
        return sorted(self._connected)

    async def run(self) -> None:
        for adapter in self._adapters:
            task = asyncio.create_task(self._run_adapter(adapter))
            self._tasks.append(task)

    async def stop(self) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()

        for adapter in self._adapters:
            try:
                await asyncio.wait_for(adapter.stop(), timeout=self._stop_timeout)
            except asyncio.TimeoutError:
                logger.error(
                    f"Adapter {adapter.name} stop timed out ({self._stop_timeout}s)"
                )
            except Exception as e:
                logger.error(f"Error stopping adapter {adapter.name}: {e}")

    async def _run_adapter(self, adapter: PlatformAdapter) -> None:
        backoff = self._initial_backoff
        while True:
            try:
                await asyncio.wait_for(
                    adapter.start(on_message=self._on_message),
                    timeout=self._start_timeout,
                )
                self._connected.add(adapter.name)
                logger.info(f"Adapter {adapter.name} connected")
                backoff = self._initial_backoff
                return
            except asyncio.CancelledError:
                return
            except asyncio.TimeoutError:
                logger.error(
                    f"Adapter {adapter.name} timed out ({self._start_timeout}s)"
                )
            except Exception as e:
                logger.error(f"Adapter {adapter.name} failed: {e}")

            self._connected.discard(adapter.name)
            logger.info(f"Adapter {adapter.name} reconnecting in {backoff:.0f}s")
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return
            backoff = min(backoff * 2, self._max_backoff)
