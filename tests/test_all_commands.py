"""Test that all CLI commands import and dispatch without crashing."""

import importlib

from cliver.command_router import HANDLERS


def test_all_commands_import():
    """Every registered command module imports and has dispatch()."""
    for cmd in sorted(HANDLERS.keys()):
        mod = importlib.import_module(HANDLERS[cmd])
        assert hasattr(mod, "dispatch"), f"{cmd} missing dispatch()"


def test_help_lists_commands(test_cliver):
    from cliver.commands.help_cmd import dispatch

    dispatch(test_cliver, "")


def test_clear_without_session(test_cliver):
    from cliver.commands.clear_cmd import dispatch

    dispatch(test_cliver, "")


def test_config_show(test_cliver):
    from cliver.commands.config import dispatch

    try:
        dispatch(test_cliver, "show")
    except SystemExit:
        pass


def test_cost_summary(test_cliver):
    from cliver.commands.cost import dispatch

    dispatch(test_cliver, "session")


def test_identity_show(test_cliver):
    from cliver.commands.identity import dispatch

    dispatch(test_cliver, "show")


def test_memory_show(test_cliver):
    from cliver.commands.memory import dispatch

    dispatch(test_cliver, "show")


def test_profile_show(test_cliver):
    from cliver.commands.profile import dispatch

    dispatch(test_cliver, "show")


def test_session_list(test_cliver):
    from cliver.commands.session_cmd import dispatch

    try:
        dispatch(test_cliver, "list")
    except SystemExit:
        pass


def test_model_list(test_cliver):
    from cliver.commands.model import dispatch

    try:
        dispatch(test_cliver, "list")
    except SystemExit:
        pass


def test_task_list(test_cliver):
    from cliver.commands.task import dispatch

    try:
        dispatch(test_cliver, "list")
    except SystemExit:
        pass


def test_skills_list(test_cliver):
    from cliver.commands.skills import dispatch

    try:
        dispatch(test_cliver, "list")
    except SystemExit:
        pass


def test_provider_list(test_cliver):
    from cliver.commands.provider import dispatch

    try:
        dispatch(test_cliver, "list")
    except SystemExit:
        pass


def test_keys_list(test_cliver):
    from cliver.commands.keys import dispatch

    try:
        dispatch(test_cliver, "list")
    except SystemExit:
        pass


def test_mcp_list(test_cliver):
    from cliver.commands.mcp import dispatch

    try:
        dispatch(test_cliver, "list")
    except SystemExit:
        pass


def test_permissions_show(test_cliver):
    from cliver.commands.permissions import dispatch

    try:
        dispatch(test_cliver, "show")
    except SystemExit:
        pass
