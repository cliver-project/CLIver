"""Tests for WeChat adapter -- formatting and config."""

from cliver.config import PlatformConfig
from cliver.gateway.adapters.wechat import WeChatAdapter, strip_markdown


class TestWeChatMarkdownStripping:
    def test_strip_bold(self):
        result = strip_markdown("**bold text**")
        assert "bold text" in result
        assert "**" not in result

    def test_strip_inline_code(self):
        result = strip_markdown("`code`")
        assert "code" in result
        assert "`" not in result

    def test_strip_code_block(self):
        result = strip_markdown("```python\nprint('hello')\n```")
        assert "print('hello')" in result
        assert "```" not in result

    def test_strip_links(self):
        result = strip_markdown("[click here](https://example.com)")
        assert "click here" in result
        assert "https://example.com" not in result

    def test_strip_headings(self):
        result = strip_markdown("## Heading\ntext")
        assert "Heading" in result
        assert "##" not in result

    def test_plain_text_unchanged(self):
        result = strip_markdown("hello world")
        assert result == "hello world"

    def test_mixed_formatting(self):
        result = strip_markdown("**bold** and `code` and _italic_")
        assert "bold" in result
        assert "code" in result
        assert "italic" in result
        assert "**" not in result
        assert "`" not in result
        assert "_" not in result


class TestWeChatAdapter:
    def test_adapter_name(self):
        cfg = PlatformConfig(type="wechat", token="fake", corp_id="corp1", agent_id="1000001")
        adapter = WeChatAdapter(cfg)
        assert adapter.name == "wechat"

    def test_max_length(self):
        cfg = PlatformConfig(type="wechat", token="fake")
        adapter = WeChatAdapter(cfg)
        assert adapter.max_message_length() == 2048

    def test_format_strips_markdown(self):
        cfg = PlatformConfig(type="wechat", token="fake")
        adapter = WeChatAdapter(cfg)
        result = adapter.format_message("**bold** and `code`")
        assert "**" not in result
        assert "`" not in result
        assert "bold" in result
        assert "code" in result

    def test_allowed_users_none(self):
        cfg = PlatformConfig(type="wechat", token="fake")
        adapter = WeChatAdapter(cfg)
        assert adapter._is_allowed("any_user") is True

    def test_allowed_users_whitelist(self):
        cfg = PlatformConfig(type="wechat", token="fake", allowed_users=["user1", "user2"])
        adapter = WeChatAdapter(cfg)
        assert adapter._is_allowed("user1") is True
        assert adapter._is_allowed("user3") is False

    def test_corp_id_stored(self):
        cfg = PlatformConfig(type="wechat", token="secret", corp_id="ww123")
        adapter = WeChatAdapter(cfg)
        assert adapter._corp_id == "ww123"

    def test_agent_id_stored(self):
        cfg = PlatformConfig(type="wechat", token="secret", agent_id="1000002")
        adapter = WeChatAdapter(cfg)
        assert adapter._agent_id == "1000002"

    def test_home_channel(self):
        cfg = PlatformConfig(type="wechat", token="fake", home_channel="user123")
        adapter = WeChatAdapter(cfg)
        assert adapter._home_channel == "user123"
