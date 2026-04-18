import time

import pytest

from cliver.llm.rate_limiter import RateLimiter, parse_period


class TestParsePeriod:
    def test_integer(self):
        assert parse_period(3600) == 3600

    def test_float(self):
        assert parse_period(3600.5) == 3600

    def test_string_seconds(self):
        assert parse_period("3600") == 3600

    def test_hours(self):
        assert parse_period("5h") == 18000

    def test_minutes(self):
        assert parse_period("30m") == 1800

    def test_days(self):
        assert parse_period("1d") == 86400

    def test_combined(self):
        assert parse_period("1d2h30m") == 86400 + 7200 + 1800

    def test_seconds_suffix(self):
        assert parse_period("90s") == 90

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid period format"):
            parse_period("abc")

    def test_zero_period(self):
        with pytest.raises(ValueError, match="must be positive"):
            parse_period("0h")


class TestRateLimiter:
    def test_unconfigured_provider_no_wait(self):
        rl = RateLimiter()
        assert rl.get_wait_time("unknown") == 0.0

    def test_configure(self):
        rl = RateLimiter()
        rl.configure("minimax", requests=100, period_seconds=100, margin=0.0)
        assert rl.is_configured("minimax")
        assert not rl.is_configured("other")

    def test_interval_with_margin(self):
        rl = RateLimiter()
        rl.configure("p", requests=100, period_seconds=100, margin=0.1)
        assert rl.get_wait_time("p") == 0.0

    def test_invalid_config_ignored(self):
        rl = RateLimiter()
        rl.configure("p", requests=0, period_seconds=100)
        assert not rl.is_configured("p")

    @pytest.mark.asyncio
    async def test_wait_records_call(self):
        rl = RateLimiter()
        rl.configure("p", requests=1000, period_seconds=1000, margin=0.0)
        waited = await rl.wait("p")
        assert waited == 0.0
        assert rl.get_wait_time("p") > 0

    @pytest.mark.asyncio
    async def test_wait_paces_calls(self):
        rl = RateLimiter()
        rl.configure("p", requests=10, period_seconds=1, margin=0.0)
        start = time.monotonic()
        await rl.wait("p")
        await rl.wait("p")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08

    @pytest.mark.asyncio
    async def test_margin_increases_interval(self):
        rl = RateLimiter()
        rl.configure("p", requests=10, period_seconds=1, margin=1.0)
        start = time.monotonic()
        await rl.wait("p")
        await rl.wait("p")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.18

    @pytest.mark.asyncio
    async def test_independent_providers(self):
        rl = RateLimiter()
        rl.configure("fast", requests=100, period_seconds=1, margin=0.0)
        rl.configure("slow", requests=1, period_seconds=10, margin=0.0)
        await rl.wait("fast")
        await rl.wait("slow")
        assert rl.get_wait_time("fast") < rl.get_wait_time("slow")

    def test_record_call(self):
        rl = RateLimiter()
        rl.configure("p", requests=10, period_seconds=10, margin=0.0)
        rl.record_call("p")
        assert rl.get_wait_time("p") > 0
