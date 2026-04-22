"""Tests for UIBridge protocol, FieldSpec, and CLIBridge."""
import threading
from unittest.mock import patch

from cliver.ui_bridge import CLIBridge, FieldSpec, TUIBridge


class TestFieldSpec:
    def test_fieldspec_defaults(self):
        f = FieldSpec(key="name", label="Name")
        assert f.key == "name"
        assert f.label == "Name"
        assert f.help == ""
        assert f.required is False
        assert f.choices is None
        assert f.default is None
        assert f.secret is False

    def test_fieldspec_all_fields(self):
        f = FieldSpec(
            key="token",
            label="API Token",
            help="Get from dashboard",
            required=True,
            choices=None,
            default="old-token",
            secret=True,
        )
        assert f.required is True
        assert f.secret is True
        assert f.default == "old-token"


class TestCLIBridge:
    def test_output(self, capsys):
        bridge = CLIBridge()
        bridge.output("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_ask_input_freeform(self):
        bridge = CLIBridge()
        with patch("builtins.input", return_value="my answer"):
            result = bridge.ask_input("Enter name: ")
        assert result == "my answer"

    def test_ask_input_with_choices_valid(self):
        bridge = CLIBridge()
        with patch("builtins.input", return_value="b"):
            result = bridge.ask_input("Pick: ", choices=["a", "b", "c"])
        assert result == "b"

    def test_ask_input_with_choices_retries(self):
        bridge = CLIBridge()
        with patch("builtins.input", side_effect=["x", "b"]):
            result = bridge.ask_input("Pick: ", choices=["a", "b", "c"])
        assert result == "b"

    def test_ask_input_eof_returns_empty(self):
        bridge = CLIBridge()
        with patch("builtins.input", side_effect=EOFError):
            result = bridge.ask_input("Enter: ")
        assert result == ""

    def test_ask_input_keyboard_interrupt_returns_empty(self):
        bridge = CLIBridge()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = bridge.ask_input("Enter: ")
        assert result == ""


class TestCLIBridgeAskFields:
    def test_ask_fields_collects_values(self):
        bridge = CLIBridge()
        fields = [
            FieldSpec(key="name", label="Name", required=True),
            FieldSpec(key="color", label="Color", choices=["red", "blue"]),
        ]
        with patch("builtins.input", side_effect=["Alice", "blue"]):
            result = bridge.ask_fields(fields)
        assert result == {"name": "Alice", "color": "blue"}

    def test_ask_fields_uses_default(self):
        bridge = CLIBridge()
        fields = [FieldSpec(key="name", label="Name", default="Bob")]
        with patch("builtins.input", return_value=""):
            result = bridge.ask_fields(fields)
        assert result == {"name": "Bob"}

    def test_ask_fields_required_cancel(self):
        bridge = CLIBridge()
        fields = [FieldSpec(key="token", label="Token", required=True)]
        with patch("builtins.input", side_effect=EOFError):
            result = bridge.ask_fields(fields)
        assert result is None

    def test_ask_fields_optional_skipped(self):
        bridge = CLIBridge()
        fields = [FieldSpec(key="channel", label="Channel")]
        with patch("builtins.input", return_value=""):
            result = bridge.ask_fields(fields)
        assert result == {}


class TestCLIBridgePermission:
    def test_ask_permission_allow(self):
        bridge = CLIBridge()
        with patch("builtins.input", return_value="y"):
            result = bridge.ask_permission("run_shell", {"cmd": "ls"}, {})
        assert result == "allow"

    def test_ask_permission_deny(self):
        bridge = CLIBridge()
        with patch("builtins.input", return_value="n"):
            result = bridge.ask_permission("run_shell", {"cmd": "rm -rf"}, {})
        assert result == "deny"

    def test_ask_permission_always(self):
        bridge = CLIBridge()
        with patch("builtins.input", return_value="a"):
            result = bridge.ask_permission("read_file", {"path": "x"}, {})
        assert result == "allow_always"

    def test_ask_permission_deny_always(self):
        bridge = CLIBridge()
        with patch("builtins.input", return_value="d"):
            result = bridge.ask_permission("run_shell", {"cmd": "x"}, {})
        assert result == "deny_always"

    def test_ask_permission_eof_denies(self):
        bridge = CLIBridge()
        with patch("builtins.input", side_effect=EOFError):
            result = bridge.ask_permission("run_shell", {"cmd": "x"}, {})
        assert result == "deny"


class TestTUIBridge:
    def test_output(self, capsys):
        bridge = TUIBridge()
        bridge.output("hello TUI")
        captured = capsys.readouterr()
        assert "hello TUI" in captured.out

    def test_ask_input_blocks_and_returns(self):
        bridge = TUIBridge()
        result = [None]

        def worker():
            result[0] = bridge.ask_input("Name: ")

        t = threading.Thread(target=worker)
        t.start()
        import time

        time.sleep(0.05)
        assert bridge._pending is not None
        bridge.receive_input("Alice")
        t.join(timeout=1)
        assert result[0] == "Alice"
        assert bridge._pending is None

    def test_ask_input_with_choices_validates(self):
        bridge = TUIBridge()
        result = [None]

        def worker():
            result[0] = bridge.ask_input("Pick: ", choices=["a", "b"])

        t = threading.Thread(target=worker)
        t.start()
        import time

        time.sleep(0.05)
        assert not bridge.try_receive("x")  # rejected
        assert bridge.try_receive("b")  # accepted
        t.join(timeout=1)
        assert result[0] == "b"

    def test_ask_permission(self):
        bridge = TUIBridge()
        result = [None]

        def worker():
            result[0] = bridge.ask_permission("bash", {"cmd": "ls"}, {})

        t = threading.Thread(target=worker)
        t.start()
        import time

        time.sleep(0.05)
        bridge.try_receive("y")
        t.join(timeout=1)
        assert result[0] == "allow"

    def test_cancel_pending(self):
        bridge = TUIBridge()
        result = [None]

        def worker():
            result[0] = bridge.ask_input("Name: ")

        t = threading.Thread(target=worker)
        t.start()
        import time

        time.sleep(0.05)
        bridge.cancel_pending()
        t.join(timeout=1)
        assert result[0] == ""

    def test_ask_fields_via_tui(self):
        bridge = TUIBridge()
        fields = [FieldSpec(key="name", label="Name", required=True)]
        result = [None]

        def worker():
            result[0] = bridge.ask_fields(fields)

        t = threading.Thread(target=worker)
        t.start()
        import time

        time.sleep(0.05)
        bridge.receive_input("Alice")
        t.join(timeout=1)
        assert result[0] == {"name": "Alice"}
