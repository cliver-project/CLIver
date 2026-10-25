"""Filesystem locations used by the CLIver application."""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from pathlib import Path

CONFIG_DIR_ENV = "CLIVER_CONF_DIR"
APP_NAME = "cliver"

__all__ = ["APP_NAME", "CONFIG_DIR_ENV", "get_config_dir"]


def get_config_dir(
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    system: str | None = None,
) -> Path:
    """Return CLIver's configuration directory.

    ``CLIVER_CONF_DIR`` overrides the platform default.  On Windows that
    default is ``%APPDATA%\\cliver`` (falling back to the roaming profile
    under *home*); everywhere else it is ``~/.cliver``.
    """
    env = os.environ if env is None else env
    override = env.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()

    home = Path.home() if home is None else Path(home).expanduser()
    system = platform.system() if system is None else system
    if system == "Windows":
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / APP_NAME
    return home / f".{APP_NAME}"
