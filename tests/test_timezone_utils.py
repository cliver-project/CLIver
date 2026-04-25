"""Tests for timezone utilities in cliver.util."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from cliver.util import configure_timezone, format_datetime, get_effective_timezone


class TestConfigureTimezone:
    def teardown_method(self):
        configure_timezone(None)

    def test_default_is_system_local(self):
        configure_timezone(None)
        tz = get_effective_timezone()
        local_tz = datetime.now().astimezone().tzinfo
        assert str(tz) == str(local_tz)

    def test_configured_timezone_overrides_local(self):
        configure_timezone("America/New_York")
        tz = get_effective_timezone()
        assert str(tz) == "America/New_York"

    def test_reset_to_none_restores_local(self):
        configure_timezone("Europe/London")
        assert str(get_effective_timezone()) == "Europe/London"
        configure_timezone(None)
        local_tz = datetime.now().astimezone().tzinfo
        assert str(get_effective_timezone()) == str(local_tz)


class TestFormatDatetime:
    def teardown_method(self):
        configure_timezone(None)

    def test_format_now_default(self):
        result = format_datetime()
        assert len(result) >= 16  # "YYYY-MM-DD HH:MM"

    def test_format_specific_datetime_in_configured_tz(self):
        configure_timezone("UTC")
        dt = datetime(2026, 4, 25, 9, 0, 0, tzinfo=timezone.utc)
        result = format_datetime(dt)
        assert result == "2026-04-25 09:00"

    def test_format_converts_to_configured_tz(self):
        configure_timezone("Asia/Shanghai")
        dt = datetime(2026, 4, 25, 1, 0, 0, tzinfo=timezone.utc)
        result = format_datetime(dt)
        # UTC 01:00 = Shanghai 09:00 (UTC+8)
        assert result == "2026-04-25 09:00"

    def test_custom_format(self):
        configure_timezone("UTC")
        dt = datetime(2026, 4, 25, 9, 30, 45, tzinfo=timezone.utc)
        result = format_datetime(dt, fmt="%H:%M:%S")
        assert result == "09:30:45"

    def test_naive_now_uses_utc_internally(self):
        configure_timezone("UTC")
        result = format_datetime()
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        assert result == now_utc
