"""Tests for Gateway message handling pipeline."""

from cliver.gateway.platform_adapter import MessageEvent


class TestSessionKeyResolution:
    def test_dm_with_thread(self):
        from cliver.gateway.gateway import Gateway

        event = MessageEvent(
            platform="telegram",
            channel_id="123",
            user_id="456",
            text="hello",
            thread_id="t1",
            message_id="m1",
        )
        key = Gateway._resolve_session_key(event)
        assert key == "telegram:123:t1"

    def test_dm_top_level_uses_message_id(self):
        from cliver.gateway.gateway import Gateway

        event = MessageEvent(
            platform="telegram",
            channel_id="123",
            user_id="456",
            text="hello",
            message_id="m99",
        )
        key = Gateway._resolve_session_key(event)
        assert key == "telegram:123:m99"

    def test_group_with_thread(self):
        from cliver.gateway.gateway import Gateway

        event = MessageEvent(
            platform="telegram",
            channel_id="-100123",
            user_id="456",
            text="hello",
            thread_id="t2",
            is_group=True,
        )
        key = Gateway._resolve_session_key(event)
        assert key == "telegram:-100123:t2"

    def test_no_thread_no_message_id(self):
        from cliver.gateway.gateway import Gateway

        event = MessageEvent(
            platform="discord",
            channel_id="chan1",
            user_id="user1",
            text="hi",
        )
        key = Gateway._resolve_session_key(event)
        assert key == "discord:chan1:"


class TestAdapterClassResolution:
    def test_load_builtin_adapter_class(self):
        from cliver.gateway.gateway import Gateway

        cls_path = Gateway._resolve_adapter_class("telegram")
        assert cls_path == "cliver.gateway.adapters.telegram.TelegramAdapter"

    def test_load_custom_adapter_class(self):
        from cliver.gateway.gateway import Gateway

        cls_path = Gateway._resolve_adapter_class("my_module.MyAdapter")
        assert cls_path == "my_module.MyAdapter"

    def test_unknown_builtin_raises(self):
        import pytest

        from cliver.gateway.gateway import Gateway

        with pytest.raises(ValueError, match="Unknown adapter type"):
            Gateway._resolve_adapter_class("whatsapp")
