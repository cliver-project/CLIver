"""Tests for Discord adapter -- formatting and config."""


class TestDiscordFormatting:
    def test_format_message_passthrough(self):
        from cliver.config import PlatformConfig
        from cliver.gateway.adapters.discord import DiscordAdapter

        cfg = PlatformConfig(type="discord", token="fake")
        adapter = DiscordAdapter(cfg)
        result = adapter.format_message("**bold** and `code`")
        assert "**bold**" in result
        assert "`code`" in result


class TestDiscordMaxLength:
    def test_max_length_is_2000(self):
        from cliver.config import PlatformConfig
        from cliver.gateway.adapters.discord import DiscordAdapter

        cfg = PlatformConfig(type="discord", token="fake")
        adapter = DiscordAdapter(cfg)
        assert adapter.max_message_length() == 2000


class TestDiscordConfig:
    def test_adapter_name(self):
        from cliver.config import PlatformConfig
        from cliver.gateway.adapters.discord import DiscordAdapter

        cfg = PlatformConfig(type="discord", token="fake")
        adapter = DiscordAdapter(cfg)
        assert adapter.name == "discord"
