"""
Profile management commands.

View and update the CLIver identity profile (name, role, preferences).
Identity is stored as YAML frontmatter in identity.md.
"""

import os
import subprocess

import click
import yaml

from cliver.agent_profile import _parse_frontmatter
from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help


def _show_profile(cliver: Cliver):
    """Display the current profile."""
    profile = cliver.agent_profile
    content = profile.load_identity()

    if not content:
        cliver.output("No profile set. Use /profile set <key> <value> to configure.")
        return

    data = profile.load_profile()
    if data:
        cliver.output("Profile")
        for key, value in data.items():
            if isinstance(value, dict):
                cliver.output(f"  {key}:")
                for k, v in value.items():
                    cliver.output(f"    {k}: {v}")
            else:
                cliver.output(f"  {key}: {value}")

        _, body = _parse_frontmatter(content)
        if body:
            cliver.output("")
            cliver.output(body)
    else:
        cliver.output(content)


def _set_field(cliver: Cliver, key: str, value: str):
    """Set a profile field in identity.md frontmatter."""
    profile = cliver.agent_profile

    # Strip surrounding quotes (user may type: /profile set name "CLIver")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]

    # Try to parse value as YAML (handles booleans, numbers, lists)
    try:
        parsed = yaml.safe_load(value)
        if isinstance(parsed, (bool, int, float)):
            value = parsed
    except yaml.YAMLError:
        pass

    profile.set_profile_field(key, value)
    cliver.output(f"Set {key} = {value}")


def _edit_profile(cliver: Cliver):
    """Open identity.md in $EDITOR."""
    profile = cliver.agent_profile
    if not profile.identity_file.exists():
        from cliver.agent_profile import _DEFAULT_IDENTITY

        profile.save_identity(_DEFAULT_IDENTITY)

    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.run([editor, str(profile.identity_file)], check=True)
        cliver.output(f"Saved {profile.identity_file}")
    except Exception as e:
        cliver.output(f"Error: {e}")


# TUI dispatch entry point

_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """View or update your CLIver profile (name, role, preferences)."""
    parts = args.strip().split(None, 2) if args.strip() else []
    sub = parts[0] if parts else "show"

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(profile_group, "/profile"))
        return

    if sub in _SUBCOMMANDS and wants_help(parts[1] if len(parts) > 1 else ""):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/profile {sub}"))
        return

    if sub == "show":
        _show_profile(cliver)
    elif sub == "set":
        if len(parts) < 3:
            cliver.output("Usage: /profile set <key> <value>")
            cliver.output("Examples:")
            cliver.output("  /profile set name Alice")
            cliver.output("  /profile set role 'Senior Backend Engineer'")
            cliver.output("  /profile set preferences.language zh-CN")
            return
        _set_field(cliver, parts[1], parts[2])
    elif sub == "edit":
        _edit_profile(cliver)
    else:
        cliver.output(f"Unknown: /profile {sub}")
        cliver.output("Available: show, set, edit")


# Click wrappers


@click.group(
    name="profile",
    help="View or update your CLIver profile (name, role, preferences)",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def profile_group(ctx, cliver: Cliver):
    if ctx.invoked_subcommand is None:
        _show_profile(cliver)


@profile_group.command(name="show", help="Display the current profile")
@pass_cliver
def show(cliver: Cliver):
    _show_profile(cliver)


@profile_group.command(name="set", help="Set a profile field (e.g., /profile set name Alice)")
@click.argument("key")
@click.argument("value")
@pass_cliver
def set_field(cliver: Cliver, key: str, value: str):
    _set_field(cliver, key, value)


@profile_group.command(name="edit", help="Open identity.md in $EDITOR for free-form editing")
@pass_cliver
def edit(cliver: Cliver):
    _edit_profile(cliver)


_SUBCOMMANDS.update(
    {
        "show": show,
        "set": set_field,
        "edit": edit,
    }
)

# Module-level alias for auto-discovery (filename stem must match)
profile = profile_group
