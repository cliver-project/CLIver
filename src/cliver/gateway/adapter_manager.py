"""Adapter lifecycle manager with crash isolation and exponential backoff."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Optional

from cliver.gateway.platform_adapter import PlatformAdapter

logger = logging.getLogger(__name__)

MessageCallback = Callable[..., Coroutine[Any, Any, None]]


@dataclass
class AdapterStatus:
    """Live status of a single adapter."""

    name: str
    state: str = "pending"  # pending, connecting, connected, error
    error: str = ""


class AdapterManager:
    """Manages platform adapter connections with crash-isolated lifecycle.

    Each adapter runs in its own asyncio Task.  Failures in one adapter
    never affect others or the gateway event loop.

    Adapters whose ``start()`` returns quickly (non-persistent) are
    treated as one-shot; adapters whose ``start()`` blocks (WebSocket
    listeners) are monitored until the connection drops.
    """

    def __init__(
        self,
        adapters: List[PlatformAdapter],
        on_message: MessageCallback,
        on_reconnect: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        initial_backoff: float = 5.0,
        max_backoff: float = 300.0,
        handshake_grace: float = 0.5,
        stop_timeout: float = 10.0,
    ):
        self._adapters = adapters
        self._on_message = on_message
        self._on_reconnect = on_reconnect
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._handshake_grace = handshake_grace
        self._stop_timeout = stop_timeout
        self._tasks: List[asyncio.Task] = []
        self._stop_events: dict[str, asyncio.Event] = {}
        self._connected: set[str] = set()
        self._statuses: dict[str, AdapterStatus] = {a.name: AdapterStatus(name=a.name) for a in adapters}

    @property
    def connected_platforms(self) -> list[str]:
        return sorted(self._connected)

    @property
    def platform_statuses(self) -> list[dict]:
        return [{"name": s.name, "state": s.state, "error": s.error} for s in self._statuses.values()]

    async def run(self) -> None:
        for adapter in self._adapters:
            stop_evt = asyncio.Event()
            self._stop_events[adapter.name] = stop_evt
            task = asyncio.create_task(self._run_adapter(adapter, stop_evt), name=f"adapter:{adapter.name}")
            self._tasks.append(task)

    async def stop(self) -> None:
        for evt in self._stop_events.values():
            evt.set()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            _, pending = await asyncio.wait(self._tasks, timeout=self._stop_timeout)
            for task in pending:
                logger.warning("Adapter task %s did not stop within %ss", task.get_name(), self._stop_timeout)
                task.cancel()

        self._tasks.clear()
        self._stop_events.clear()

        for adapter in self._adapters:
            try:
                await asyncio.wait_for(adapter.stop(), timeout=self._stop_timeout)
            except asyncio.TimeoutError:
                logger.error("Adapter %s stop timed out (%ss)", adapter.name, self._stop_timeout)
            except Exception:
                logger.exception("Error stopping adapter %s", adapter.name)

    async def _run_adapter(self, adapter: PlatformAdapter, stop_evt: asyncio.Event) -> None:
        """Run adapter with crash isolation and exponential backoff.

        Handles two adapter patterns:
        - **Non-persistent**: ``start()`` returns after connecting (tests, simple adapters).
          Reconnects are handled by looping.
        - **Persistent**: ``start()`` blocks indefinitely (WebSocket listeners).
          The task is monitored and reconnected on drop.
        """
        status = self._statuses[adapter.name]
        backoff = self._initial_backoff

        while not stop_evt.is_set():
            adapter_task: Optional[asyncio.Task] = None
            try:
                status.state = "connecting"
                status.error = ""

                adapter_task = asyncio.create_task(
                    adapter.start(on_message=self._on_message),
                    name=f"adapter-run:{adapter.name}",
                )

                # Wait briefly to see if start() returns (non-persistent adapter).
                stop_task = asyncio.ensure_future(stop_evt.wait())
                done, _ = await asyncio.wait(
                    [adapter_task, stop_task],
                    timeout=self._handshake_grace,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not stop_task.done():
                    stop_task.cancel()

                if stop_evt.is_set():
                    _cancel_and_wait(adapter_task)
                    return

                if adapter_task.done():
                    # Non-persistent: start() returned (success or failure)
                    exc = adapter_task.exception()
                    if exc is None:
                        await self._on_connected(adapter.name, status)
                        backoff = self._initial_backoff
                        await stop_evt.wait()
                        return
                    raise exc

                # Persistent: start() is still running (WebSocket listener).
                await self._on_connected(adapter.name, status)
                backoff = self._initial_backoff

                # Wait for connection drop or stop signal
                stop_task2 = asyncio.ensure_future(stop_evt.wait())
                done, _ = await asyncio.wait(
                    [adapter_task, stop_task2],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not stop_task2.done():
                    stop_task2.cancel()

                if stop_evt.is_set():
                    _cancel_and_wait(adapter_task)
                    return

                exc = adapter_task.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    raise exc

            except asyncio.CancelledError:
                _cancel_and_wait(adapter_task)
                return
            except Exception as e:
                status.state = "error"
                status.error = str(e)
                logger.exception("Adapter %s failed: %s", adapter.name, e)
            finally:
                self._connected.discard(adapter.name)
                if adapter_task and not adapter_task.done():
                    _cancel_and_wait(adapter_task)

            if stop_evt.is_set():
                return

            logger.info("Adapter %s reconnecting in %.0fs", adapter.name, backoff)
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, self._max_backoff)

    async def _on_connected(self, name: str, status: AdapterStatus) -> None:
        """Mark adapter as connected and fire reconnect callback (with crash isolation)."""
        self._connected.add(name)
        status.state = "connected"
        status.error = ""
        logger.info("Adapter %s connected", name)
        if self._on_reconnect:
            try:
                await self._on_reconnect(name)
            except Exception:
                logger.exception("Reconnect callback failed for %s", name)


def _cancel_and_wait(task: Optional[asyncio.Task]) -> None:
    if task is None or task.done():
        return
    task.cancel()
