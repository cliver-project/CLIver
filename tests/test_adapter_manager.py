"""Tests for AdapterManager with reconnect logic."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from cliver.gateway.adapter_manager import AdapterManager


class FakeAdapter:
    def __init__(self, name, fail_count=0):
        self.name = name
        self._fail_count = fail_count
        self._attempts = 0
        self.started = False
        self.stopped = False

    async def start(self, on_message=None):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise ConnectionError(f"{self.name} failed")
        self.started = True

    async def stop(self):
        self.stopped = True


class TestAdapterManager:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        adapter = FakeAdapter("test")
        mgr = AdapterManager([adapter], on_message=AsyncMock())
        await mgr.run()
        await asyncio.sleep(0.05)
        assert adapter.started
        assert "test" in mgr.connected_platforms
        await mgr.stop()
        assert adapter.stopped

    @pytest.mark.asyncio
    async def test_reconnect_on_failure(self):
        adapter = FakeAdapter("flaky", fail_count=2)
        mgr = AdapterManager(
            [adapter], on_message=AsyncMock(), initial_backoff=0.01, max_backoff=0.05
        )
        await mgr.run()
        await asyncio.sleep(0.2)
        assert adapter.started
        assert adapter._attempts == 3
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_multiple_adapters_independent(self):
        good = FakeAdapter("good")
        bad = FakeAdapter("bad", fail_count=100)
        mgr = AdapterManager(
            [good, bad],
            on_message=AsyncMock(),
            initial_backoff=0.01,
            max_backoff=0.05,
        )
        await mgr.run()
        await asyncio.sleep(0.05)
        assert good.started
        assert not bad.started
        assert "good" in mgr.connected_platforms
        assert "bad" not in mgr.connected_platforms
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_reconnect(self):
        adapter = FakeAdapter("slow", fail_count=100)
        mgr = AdapterManager([adapter], on_message=AsyncMock(), initial_backoff=10)
        await mgr.run()
        await asyncio.sleep(0.05)
        await mgr.stop()
        assert adapter._attempts <= 2

    @pytest.mark.asyncio
    async def test_connected_platforms_empty_initially(self):
        mgr = AdapterManager([], on_message=AsyncMock())
        assert mgr.connected_platforms == []

    @pytest.mark.asyncio
    async def test_stop_with_timeout(self):
        class HangingAdapter:
            name = "hanger"
            started = False

            async def start(self, on_message=None):
                self.started = True

            async def stop(self):
                await asyncio.sleep(100)

        adapter = HangingAdapter()
        mgr = AdapterManager([adapter], on_message=AsyncMock(), stop_timeout=0.1)
        await mgr.run()
        await asyncio.sleep(0.05)
        await mgr.stop()
