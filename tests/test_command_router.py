"""Tests for CommandRouter."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from cliver.command_router import HANDLERS, CommandRouter


class MockCliver:
    def __init__(self):
        self.outputs = []
        self.session_options = {}
        self.conversation_messages = []
        self.task_executor = None
        self.ui = MagicMock()
        self.ui.output = lambda text, **kw: self.outputs.append(text)
        self._cancel_requested = False

    def output(self, *args, **kwargs):
        self.outputs.append(args[0] if args else "")

    def record_turn(self, role, content):
        pass


@pytest.fixture
def cliver():
    return MockCliver()


@pytest.fixture
def router(cliver):
    return CommandRouter(cliver)


class TestHandlerRegistry:
    def test_handlers_dict_has_entries(self):
        assert "model" in HANDLERS
        assert "gateway" in HANDLERS
        assert "session" in HANDLERS
        assert "permissions" in HANDLERS

    def test_handlers_values_are_module_paths(self):
        for name, path in HANDLERS.items():
            assert path.startswith("cliver.commands."), f"{name}: {path}"


class TestCommandSync:
    def test_unknown_command(self, router, cliver):
        router.command_sync("nonexistent", "")
        assert any("Unknown command" in o for o in cliver.outputs)

    def test_known_command_dispatches(self, router):
        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_import.return_value = mock_mod
            router.command_sync("model", "list")
            mock_mod.dispatch.assert_called_once()


class TestCommandAsync:
    @pytest.mark.asyncio
    async def test_async_command_runs_in_executor(self, router):
        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_import.return_value = mock_mod
            await router.command("model", "list")
            await asyncio.sleep(0.05)
            mock_mod.dispatch.assert_called_once()


class TestBusyState:
    def test_not_busy_initially(self, router):
        assert router.is_busy is False
        assert router.is_query_active is False


class TestPendingInput:
    def test_inject_and_drain(self, router):
        router.inject_input("follow up")
        assert router.drain_pending() == "follow up"
        assert router.drain_pending() is None

    def test_drain_empty(self, router):
        assert router.drain_pending() is None
