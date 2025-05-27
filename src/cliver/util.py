import platform
import sys
import os
import select
import stat
from pathlib import Path
from cliver.constants import *


def get_config_dir() -> Path:
    """
    Returns the config directory for CLiver.
    It can be overridden by the environment of 'CLIVER_CONF_DIR'
    """
    config_dir = os.getenv(CONFIG_DIR)
    if config_dir is not None:
        return Path(config_dir)
    system = platform.system()
    if system == "Windows":
        return Path(os.getenv("APPDATA")) / APP_NAME
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        return Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


def stdin_is_piped():
    try:
        fd = sys.stdin.fileno()
        mode = os.fstat(fd).st_mode
        # True if stdin is a FIFO (pipe)
        return not os.isatty(fd) and stat.S_ISFIFO(mode)
    except:
        return True


def read_piped_input(timeout=5.0):
    """Non-blocking read from stdin if data is available."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read()
    return None
