"""
Keys management commands.

Manage encrypted keys stored in the local KeyStore.
Keys are used to resolve secrets in config.yaml (api_key, token, etc.).
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help


def _get_key_store(cliver: Cliver):
    from cliver.key_store import KeyStore

    db_path = cliver.agent_profile.db_path
    return KeyStore(db_path)


def _list_keys(cliver: Cliver):
    ks = _get_key_store(cliver)
    keys = ks.list_keys()
    if not keys:
        cliver.output("No keys stored. Use `/keys add <name>` to add one.")
        return
    lines = []
    for k in keys:
        desc = f" — {k.description}" if k.description else ""
        lines.append(f"  {k.name}{desc}")
    cliver.output("Stored keys:\n" + "\n".join(lines))


def _add_key(cliver: Cliver, name: str):
    ks = _get_key_store(cliver)
    value = cliver.ui.ask_password(f"Enter value for '{name}': ")
    if not value:
        cliver.output("Cancelled — no value entered.")
        return
    description = cliver.ui.ask_input("Description (optional): ")
    ks.set(name, value, description=description or "")
    cliver.output(f"Key '{name}' saved.")


def _remove_key(cliver: Cliver, name: str):
    ks = _get_key_store(cliver)
    if ks.delete(name):
        cliver.output(f"Key '{name}' removed.")
    else:
        cliver.output(f"Key '{name}' not found.")


_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage encrypted keys — list, add, remove."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(keys_group, "/keys"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/keys {sub}"))
        return

    if sub == "list":
        _list_keys(cliver)
    elif sub == "add":
        if not rest:
            cliver.output("Usage: /keys add <name>")
            return
        _add_key(cliver, rest)
    elif sub == "remove":
        if not rest:
            cliver.output("Usage: /keys remove <name>")
            return
        _remove_key(cliver, rest)
    else:
        cliver.output(f"Unknown: /keys {sub}")
        cliver.output("Available: list, add, remove")


@click.group(
    name="keys",
    help="Manage encrypted keys — list, add, remove",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def keys_group(ctx, cliver: Cliver):
    if ctx.invoked_subcommand is None:
        _list_keys(cliver)


@keys_group.command(name="list", help="List all stored key names")
@pass_cliver
def list_cmd(cliver: Cliver):
    _list_keys(cliver)


@keys_group.command(name="add", help="Add or update a key")
@click.argument("name")
@pass_cliver
def add_cmd(cliver: Cliver, name: str):
    _add_key(cliver, name)


@keys_group.command(name="remove", help="Remove a key")
@click.argument("name")
@pass_cliver
def remove_cmd(cliver: Cliver, name: str):
    _remove_key(cliver, name)


keys = keys_group

_SUBCOMMANDS.update(
    {
        "list": list_cmd,
        "add": add_cmd,
        "remove": remove_cmd,
    }
)
