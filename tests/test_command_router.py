"""Tests for CommandRouter multi-slot task tracking."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from cliver.command_router import HANDLERS, CommandRouter, get_current_task_label


class MockCliver:
    def __init__(self):
        self.outputs = []
        self.session_options = {}
        self.conversation_messages = []
        self.agent_core = None
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


def _make_mock_cliver():
    """Create a minimal mock Cliver for CommandRouter tests."""
    cliver = MagicMock()
    cliver.output = MagicMock()
    cliver.session_options = {}
    cliver._cancel_requested = False
    return cliver


class TestMultiSlotTracking:
    def test_no_tasks_initially(self):
        cliver = _make_mock_cliver()
        router = CommandRouter(cliver)
        assert not router.has_active_tasks
        assert not router.has_active_query
        router.shutdown()

    def test_command_does_not_block_query_gate(self):
        """A running command should NOT set has_active_query."""
        cliver = _make_mock_cliver()
        router = CommandRouter(cliver)

        started = threading.Event()
        hold = threading.Event()

        def slow_dispatch(cli, args):
            started.set()
            hold.wait(timeout=5)

        import cliver.command_router as cr

        original_handlers = cr.HANDLERS.copy()
        cr.HANDLERS["_test"] = "cliver.commands.config"

        import cliver.commands.config as cfg

        orig_dispatch = cfg.dispatch

        cfg.dispatch = slow_dispatch
        try:
            router._dispatch_command_async("_test", "")
            started.wait(timeout=2)
            assert router.has_active_tasks
            assert not router.has_active_query
        finally:
            hold.set()
            time.sleep(0.1)
            cfg.dispatch = orig_dispatch
            cr.HANDLERS.clear()
            cr.HANDLERS.update(original_handlers)
            router.shutdown()

    def test_cancel_newest_cancels_last_task(self):
        cliver = _make_mock_cliver()
        router = CommandRouter(cliver)

        hold1 = threading.Event()
        hold2 = threading.Event()

        def task1():
            hold1.wait(timeout=5)

        def task2():
            hold2.wait(timeout=5)

        f1 = router._executor.submit(task1)
        router._register_task(f1, "test1", "command")
        f2 = router._executor.submit(task2)
        router._register_task(f2, "test2", "command")

        assert router.cancel_newest()
        hold1.set()
        hold2.set()
        time.sleep(0.1)
        router.shutdown()

    def test_task_auto_removes_on_completion(self):
        cliver = _make_mock_cliver()
        router = CommandRouter(cliver)

        done = threading.Event()

        def quick():
            done.set()

        f = router._executor.submit(quick)
        router._register_task(f, "quick", "command")
        done.wait(timeout=2)
        time.sleep(0.2)
        assert not router.has_active_tasks
        router.shutdown()


class TestTaskLabel:
    def test_label_set_during_dispatch(self):
        """Thread-local label should be set while a command runs."""
        cliver = _make_mock_cliver()
        router = CommandRouter(cliver)
        captured_label = [None]

        def capture_dispatch(cli, args):
            captured_label[0] = get_current_task_label()

        import cliver.command_router as cr

        original_handlers = cr.HANDLERS.copy()
        cr.HANDLERS["_label_test"] = "cliver.commands.config"

        import cliver.commands.config as cfg

        orig_dispatch = cfg.dispatch

        cfg.dispatch = capture_dispatch
        try:
            router.command_sync("_label_test", "")
            assert captured_label[0] == "_label_test"
        finally:
            cfg.dispatch = orig_dispatch
            cr.HANDLERS.clear()
            cr.HANDLERS.update(original_handlers)
            router.shutdown()

    def test_label_none_outside_task(self):
        assert get_current_task_label() is None


class TestOutputLabeling:
    def test_output_prefixed_for_command(self):
        from cliver.command_router import _task_context

        _task_context.label = "task:run"
        try:
            label = get_current_task_label()
            assert label == "task:run"
        finally:
            _task_context.label = None

    def test_output_no_prefix_for_chat(self):
        from cliver.command_router import _task_context

        _task_context.label = "chat"
        try:
            label = get_current_task_label()
            assert label == "chat"
        finally:
            _task_context.label = None

    def test_output_no_prefix_outside_task(self):
        assert get_current_task_label() is None


class TestClearCommand:
    def test_clear_resets_conversation(self, init_config):
        from cliver.cli import Cliver

        cliver = Cliver()
        cliver.conversation_messages = ["msg1", "msg2"]
        cliver.session_history = [{"role": "user", "content": "hi"}]
        cliver.current_session_id = "test-session"

        from cliver.commands.clear_cmd import dispatch

        dispatch(cliver, "")

        assert cliver.conversation_messages == []
        assert cliver.session_history == []
        assert cliver.current_session_id is None
