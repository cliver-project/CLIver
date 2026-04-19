"""Tests for CommandDispatcher."""

import asyncio
import pytest
from cliver.command_dispatcher import CommandDispatcher


class MockCliver:
    def __init__(self):
        self.outputs = []
        self.console = type("C", (), {"print": lambda self, *a, **kw: None})()
        self.session_options = {}
        self.conversation_messages = []
        self.task_executor = None

    def output(self, *args, **kwargs):
        self.outputs.append(args[0] if args else "")


@pytest.fixture
def cliver():
    return MockCliver()


@pytest.fixture
def dispatcher(cliver):
    return CommandDispatcher(cliver)


class TestDispatchRouting:
    @pytest.mark.asyncio
    async def test_slash_command_routes_to_handler(self, dispatcher, cliver):
        called_with = {}
        async def mock_handler(cli, args):
            called_with["cli"] = cli
            called_with["args"] = args

        dispatcher.register("test", mock_handler)
        await dispatcher.dispatch("/test some args")
        await asyncio.sleep(0.01)

        assert called_with["cli"] is cliver
        assert called_with["args"] == "some args"

    @pytest.mark.asyncio
    async def test_unknown_slash_command_shows_error(self, dispatcher, cliver):
        await dispatcher.dispatch("/nonexistent")
        assert any("Unknown command" in o for o in cliver.outputs)

    @pytest.mark.asyncio
    async def test_exit_commands(self, dispatcher):
        for cmd in ("exit", "quit", "/exit", "/quit"):
            result = await dispatcher.dispatch(cmd)
            assert result == "exit"

    @pytest.mark.asyncio
    async def test_empty_input(self, dispatcher):
        result = await dispatcher.dispatch("")
        assert result is None

    @pytest.mark.asyncio
    async def test_slash_only(self, dispatcher):
        result = await dispatcher.dispatch("/")
        assert result is None


class TestPendingInput:
    def test_drain_pending_returns_items(self, dispatcher):
        dispatcher._pending_input.append("hello")
        dispatcher._pending_input.append("world")
        assert dispatcher.drain_pending() == "hello"
        assert dispatcher.drain_pending() == "world"
        assert dispatcher.drain_pending() is None

    def test_drain_empty_returns_none(self, dispatcher):
        assert dispatcher.drain_pending() is None

    def test_is_chat_active_false_initially(self, dispatcher):
        assert dispatcher.is_chat_active is False
