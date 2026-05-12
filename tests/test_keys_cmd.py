"""Tests for /keys CLI command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cliver.key_store import KeyStore


@pytest.fixture
def mock_cliver():
    cliver = MagicMock()
    cliver.output = MagicMock()
    cliver.ui = MagicMock()
    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        cliver.agent_profile.config_dir = Path(tmpdir)
        with patch("cliver.commands.keys._get_key_store", return_value=ks):
            yield cliver, ks


def test_list_empty(mock_cliver):
    from cliver.commands.keys import dispatch

    cliver, ks = mock_cliver
    dispatch(cliver, "list")
    cliver.output.assert_called()
    output = cliver.output.call_args[0][0]
    assert "No keys" in output


def test_list_with_keys(mock_cliver):
    from cliver.commands.keys import dispatch

    cliver, ks = mock_cliver
    ks.set("openai_key", "sk-123", description="OpenAI API key")
    ks.set("anthropic_key", "sk-456")
    dispatch(cliver, "list")
    cliver.output.assert_called()
    output = cliver.output.call_args[0][0]
    assert "openai_key" in output
    assert "anthropic_key" in output


def test_add_key(mock_cliver):
    from cliver.commands.keys import dispatch

    cliver, ks = mock_cliver
    cliver.ui.ask_password = MagicMock(return_value="sk-secret-value")
    cliver.ui.ask_input = MagicMock(return_value="My API key")
    dispatch(cliver, "add my_key")
    assert ks.get("my_key") == "sk-secret-value"


def test_add_key_no_name(mock_cliver):
    from cliver.commands.keys import dispatch

    cliver, ks = mock_cliver
    dispatch(cliver, "add")
    cliver.output.assert_called()
    output = cliver.output.call_args[0][0]
    assert "Usage" in output


def test_remove_key(mock_cliver):
    from cliver.commands.keys import dispatch

    cliver, ks = mock_cliver
    ks.set("to_delete", "val")
    dispatch(cliver, "remove to_delete")
    assert ks.get("to_delete") is None


def test_remove_nonexistent(mock_cliver):
    from cliver.commands.keys import dispatch

    cliver, ks = mock_cliver
    dispatch(cliver, "remove nope")
    cliver.output.assert_called()
    output = cliver.output.call_args[0][0]
    assert "not found" in output.lower()
