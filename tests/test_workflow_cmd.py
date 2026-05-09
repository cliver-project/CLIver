"""Basic smoke tests for workflow CLI dispatch."""

from unittest.mock import MagicMock, patch


def test_dispatch_list():
    from cliver.commands.workflow_cmd import dispatch

    cliver = MagicMock()
    cliver.agent_profile.workflows_dir = "/tmp/workflows"
    with patch("cliver.commands.workflow_cmd._list_workflows") as mock_list:
        dispatch(cliver, "list")
        mock_list.assert_called_once_with(cliver)


def test_dispatch_unknown():
    from cliver.commands.workflow_cmd import dispatch

    cliver = MagicMock()
    dispatch(cliver, "foobar")
    calls = [str(c) for c in cliver.output.call_args_list]
    assert any("Unknown subcommand: foobar" in c for c in calls)
    assert any("Usage:" in c for c in calls)


def test_dispatch_empty():
    from cliver.commands.workflow_cmd import dispatch

    cliver = MagicMock()
    dispatch(cliver, "")
    calls = [str(c) for c in cliver.output.call_args_list]
    assert any("Usage:" in c for c in calls)


def test_dispatch_help():
    from cliver.commands.workflow_cmd import dispatch

    cliver = MagicMock()
    dispatch(cliver, "--help")
    calls = [str(c) for c in cliver.output.call_args_list]
    assert any("Usage:" in c for c in calls)
    assert any("delete" in c for c in calls)
