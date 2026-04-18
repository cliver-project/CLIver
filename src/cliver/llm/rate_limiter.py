"""
Per-provider rate limiter that paces LLM API calls to stay within provider limits.

Given a provider limit like "5000 requests per 5 hours", calculates the minimum
interval between calls and adds a configurable safety margin so we fire slightly
less frequently than the limit allows.
"""

import asyncio
import logging
import re
import time

logger = logging.getLogger(__name__)

_PERIOD_PATTERN = re.compile(
    r"(?:(\d+)\s*d)?\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?",
    re.IGNORECASE,
)


def parse_period(period) -> int:
    if isinstance(period, (int, float)):
        return int(period)

    s = str(period).strip()
    if s.isdigit():
        return int(s)

    m = _PERIOD_PATTERN.fullmatch(s)
    if not m or not any(m.groups()):
        raise ValueError(f"Invalid period format: {period!r}. Use e.g. '5h', '30m', '1d2h', or seconds.")

    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    minutes = int(m.group(3) or 0)
    seconds = int(m.group(4) or 0)
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError(f"Period must be positive, got: {period!r}")
    return total


class RateLimiter:
    DEFAULT_MARGIN = 0.1

    def __init__(self):
        self._last_call: dict[str, float] = {}
        self._intervals: dict[str, float] = {}

    def configure(self, provider: str, requests: int, period_seconds: int, margin: float = DEFAULT_MARGIN) -> None:
        if requests <= 0 or period_seconds <= 0:
            logger.warning("Invalid rate limit for %s: requests=%d, period=%d", provider, requests, period_seconds)
            return
        base_interval = period_seconds / requests
        self._intervals[provider] = base_interval * (1.0 + margin)
        logger.info(
            "Rate limit for %s: %d req/%ds -> %.2fs interval (%.0f%% margin)",
            provider,
            requests,
            period_seconds,
            self._intervals[provider],
            margin * 100,
        )

    def is_configured(self, provider: str) -> bool:
        return provider in self._intervals

    def get_wait_time(self, provider: str) -> float:
        if provider not in self._intervals:
            return 0.0
        interval = self._intervals[provider]
        last = self._last_call.get(provider, 0.0)
        elapsed = time.monotonic() - last
        return max(0.0, interval - elapsed)

    async def wait(self, provider: str) -> float:
        wait_time = self.get_wait_time(provider)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self._last_call[provider] = time.monotonic()
        return wait_time

    def record_call(self, provider: str) -> None:
        self._last_call[provider] = time.monotonic()
