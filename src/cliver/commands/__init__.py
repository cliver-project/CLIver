import importlib
import logging
import os
import sys
from typing import Callable, List

import click

from cliver.util import get_config_dir

logger = logging.getLogger(__name__)


def loads_commands(group: click.Group) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    _load_commands_from_dir(
        current_dir,
        group,
        package_name="cliver.commands",
        filter_fn=lambda f_name: f_name != "__init__.py",
    )


# This will load py modules from config directory
# This assumes the py modules are safe and should be set up manually.
def loads_external_commands(group: click.Group) -> None:
    config_dir = get_config_dir()
    dir_str = str(config_dir.absolute() / "commands")
    if dir_str not in sys.path:
        sys.path.append(dir_str)
    _load_commands_from_dir(dir_str, group, log=True)


def _load_commands_from_dir(
    commands_dir: str,
    group: click.Group,
    package_name: str = None,
    filter_fn: Callable[[str], bool] = None,
    log: bool = False,
) -> None:
    if commands_dir and not os.path.exists(commands_dir):
        logger.warning("Commands directory: %s does not exist", commands_dir)
        return
    for filename in os.listdir(commands_dir):
        if filename.endswith(".py"):
            # either we don't filter or filter_fn returns True
            if filter_fn is None or filter_fn(filename):
                if log:
                    full = os.path.abspath(os.path.join(commands_dir, filename))
                    logger.debug("Loads command from: %s", full)
                grp_name = filename[:-3]
                module_name = f"{grp_name}"
                if package_name is not None:
                    module_name = f"{package_name}.{grp_name}"
                module = importlib.import_module(module_name)
                if hasattr(module, grp_name):
                    cli_obj = getattr(module, grp_name)
                    if isinstance(cli_obj, click.Command):
                        group.add_command(cli_obj)
                if hasattr(module, "post_group"):
                    pg_obj = module.post_group
                    pg_obj()


def list_commands_names(group: click.Group) -> List[str]:
    return [name for name, _ in group.commands.items()]
