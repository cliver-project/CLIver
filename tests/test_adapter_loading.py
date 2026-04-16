"""Tests for adapter loading from config."""

from cliver.config import PlatformConfig


class TestPlatformConfig:
    def test_builtin_type(self):
        cfg = PlatformConfig(type="telegram", token="abc123")
        assert cfg.type == "telegram"
        assert cfg.token == "abc123"
        assert cfg.is_builtin is True

    def test_custom_type(self):
        cfg = PlatformConfig(type="my_module.MyAdapter", token="xyz")
        assert cfg.is_builtin is False

    def test_allowed_users_optional(self):
        cfg = PlatformConfig(type="telegram", token="abc")
        assert cfg.allowed_users is None

    def test_allowed_users_set(self):
        cfg = PlatformConfig(type="telegram", token="abc", allowed_users=["123"])
        assert cfg.allowed_users == ["123"]

    def test_home_channel_optional(self):
        cfg = PlatformConfig(type="telegram", token="abc")
        assert cfg.home_channel is None

    def test_extra_fields_allowed(self):
        cfg = PlatformConfig(type="telegram", token="abc", webhook_url="https://example.com")
        assert cfg.webhook_url == "https://example.com"


class TestAdapterRegistry:
    def test_builtin_registry_has_telegram(self):
        from cliver.gateway.adapters import BUILTIN_ADAPTERS
        assert "telegram" in BUILTIN_ADAPTERS

    def test_builtin_registry_has_discord(self):
        from cliver.gateway.adapters import BUILTIN_ADAPTERS
        assert "discord" in BUILTIN_ADAPTERS
