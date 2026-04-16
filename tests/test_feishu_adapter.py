"""Tests for Feishu adapter -- formatting and config."""

from cliver.config import PlatformConfig
from cliver.gateway.adapters.feishu import FeishuAdapter


class TestFeishuFormatting:
    def test_format_message_passthrough(self):
        cfg = PlatformConfig(type="feishu", token="fake")
        adapter = FeishuAdapter(cfg)
        result = adapter.format_message("**bold** and `code`")
        assert "**bold**" in result  # markdown preserved
        assert "`code`" in result

    def test_format_preserves_code_blocks(self):
        cfg = PlatformConfig(type="feishu", token="fake")
        adapter = FeishuAdapter(cfg)
        result = adapter.format_message("```python\nprint('hello')\n```")
        assert "```python" in result
        assert "print('hello')" in result


class TestFeishuAdapter:
    def test_adapter_name(self):
        cfg = PlatformConfig(type="feishu", token="fake", app_id="cli_xxx")
        adapter = FeishuAdapter(cfg)
        assert adapter.name == "feishu"

    def test_max_length(self):
        cfg = PlatformConfig(type="feishu", token="fake")
        adapter = FeishuAdapter(cfg)
        assert adapter.max_message_length() == 4000

    def test_allowed_users_none(self):
        cfg = PlatformConfig(type="feishu", token="fake")
        adapter = FeishuAdapter(cfg)
        assert adapter._is_allowed("any_user") is True

    def test_allowed_users_whitelist(self):
        cfg = PlatformConfig(type="feishu", token="fake", allowed_users=["ou_abc", "ou_def"])
        adapter = FeishuAdapter(cfg)
        assert adapter._is_allowed("ou_abc") is True
        assert adapter._is_allowed("ou_xyz") is False

    def test_app_id_stored(self):
        cfg = PlatformConfig(type="feishu", token="secret", app_id="cli_xxx")
        adapter = FeishuAdapter(cfg)
        assert adapter._app_id == "cli_xxx"

    def test_verification_token_stored(self):
        cfg = PlatformConfig(type="feishu", token="secret", verification_token="v_token")
        adapter = FeishuAdapter(cfg)
        assert adapter._verification_token == "v_token"

    def test_home_channel(self):
        cfg = PlatformConfig(type="feishu", token="fake", home_channel="oc_xxx")
        adapter = FeishuAdapter(cfg)
        assert adapter._home_channel == "oc_xxx"
