import os
import click
import importlib
from typing import List


def loads_commands(group: click.Group) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename != os.path.basename(__file__):
            grp_name = filename[:-3]
            module_name = f"cliver.commands.{grp_name}"
            module = importlib.import_module(module_name)
            if hasattr(module, grp_name):
                cli_obj = getattr(module, grp_name)
                if isinstance(cli_obj, click.Command):
                    group.add_command(cli_obj)
            if hasattr(module, "post_group"):
                pg_obj = getattr(module, "post_group")
                pg_obj()


def list_commands_names(group: click.Group) -> List[str]:
    return [name for name, _ in group.commands.items()]
