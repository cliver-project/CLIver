"""Tests for Slack adapter -- formatting and config."""

from cliver.config import PlatformConfig
from cliver.gateway.adapters.slack import SlackAdapter, markdown_to_mrkdwn


class TestSlackMrkdwn:
    def test_convert_bold(self):
        result = markdown_to_mrkdwn("**bold text**")
        assert "*bold text*" in result
        assert "**" not in result

    def test_preserve_inline_code(self):
        result = markdown_to_mrkdwn("`code` and more")
        assert "`code`" in result

    def test_preserve_code_block(self):
        result = markdown_to_mrkdwn("```python\nprint('hello')\n```")
        assert "print('hello')" in result

    def test_mixed_formatting(self):
        result = markdown_to_mrkdwn("**bold** and `code` and ```block```")
        assert "*bold*" in result
        assert "**" not in result
        assert "`code`" in result
        assert "```block```" in result

    def test_no_bold_plain_text(self):
        result = markdown_to_mrkdwn("hello world")
        assert result == "hello world"


class TestSlackAdapter:
    def test_adapter_name(self):
        cfg = PlatformConfig(type="slack", token="fake", app_token="fake-app")
        adapter = SlackAdapter(cfg)
        assert adapter.name == "slack"

    def test_max_length(self):
        cfg = PlatformConfig(type="slack", token="fake")
        adapter = SlackAdapter(cfg)
        assert adapter.max_message_length() == 4000

    def test_format_bold(self):
        cfg = PlatformConfig(type="slack", token="fake")
        adapter = SlackAdapter(cfg)
        result = adapter.format_message("**bold text**")
        assert "*bold text*" in result
        assert "**" not in result

    def test_format_code_preserved(self):
        cfg = PlatformConfig(type="slack", token="fake")
        adapter = SlackAdapter(cfg)
        result = adapter.format_message("`code` and ```block```")
        assert "`code`" in result

    def test_allowed_users_none(self):
        cfg = PlatformConfig(type="slack", token="fake")
        adapter = SlackAdapter(cfg)
        assert adapter._is_allowed("any_user") is True

    def test_allowed_users_whitelist(self):
        cfg = PlatformConfig(type="slack", token="fake", allowed_users=["U123", "U456"])
        adapter = SlackAdapter(cfg)
        assert adapter._is_allowed("U123") is True
        assert adapter._is_allowed("U789") is False

    def test_app_token_stored(self):
        cfg = PlatformConfig(type="slack", token="bot-token", app_token="app-token")
        adapter = SlackAdapter(cfg)
        assert adapter._app_token == "app-token"

    def test_home_channel(self):
        cfg = PlatformConfig(type="slack", token="fake", home_channel="C123")
        adapter = SlackAdapter(cfg)
        assert adapter._home_channel == "C123"
