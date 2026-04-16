"""Tests for Telegram adapter — formatting and message handling."""


class TestTelegramMarkdownV2:
    def test_escape_special_chars(self):
        from cliver.gateway.adapters.telegram import escape_markdown_v2

        result = escape_markdown_v2("hello.world!")
        assert result == r"hello\.world\!"

    def test_preserve_bold(self):
        from cliver.gateway.adapters.telegram import markdown_to_telegram

        result = markdown_to_telegram("**bold text**")
        assert "*bold text*" in result

    def test_preserve_code_block(self):
        from cliver.gateway.adapters.telegram import markdown_to_telegram

        result = markdown_to_telegram("```python\nprint('hello')\n```")
        assert "print('hello')" in result
        assert r"\(" not in result

    def test_preserve_inline_code(self):
        from cliver.gateway.adapters.telegram import markdown_to_telegram

        result = markdown_to_telegram("use `foo.bar()` here")
        assert "`foo.bar()`" in result

    def test_plain_text_escaped(self):
        from cliver.gateway.adapters.telegram import markdown_to_telegram

        result = markdown_to_telegram("price is 10.5 + tax!")
        assert r"\." in result
        assert r"\+" in result
        assert r"\!" in result


class TestTelegramConfig:
    def test_adapter_name(self):
        from cliver.config import PlatformConfig

        cfg = PlatformConfig(type="telegram", token="test-token")
        assert cfg.type == "telegram"
        assert cfg.token == "test-token"


class TestTelegramMaxLength:
    def test_max_length_is_4096(self):
        from cliver.config import PlatformConfig
        from cliver.gateway.adapters.telegram import TelegramAdapter

        cfg = PlatformConfig(type="telegram", token="fake")
        adapter = TelegramAdapter(cfg)
        assert adapter.max_message_length() == 4096
