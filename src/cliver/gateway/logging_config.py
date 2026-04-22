"""Unified logging configuration for the gateway daemon.

Configures a single RotatingFileHandler on the root logger so that all
gateway components — AgentCore, adapters, scheduler, aiohttp, tools —
write to one rotating log file. Third-party library log levels are tuned
to reduce noise.
"""

import faulthandler
import logging
import sys
from logging.handlers import RotatingFileHandler

from cliver.util import get_config_dir


def configure_gateway_logging(gateway_config=None) -> None:
    """Set up logging for the gateway daemon process.

    Call this once in the forked child, before starting the aiohttp app.
    All loggers in the process will write to the rotating log file.

    Args:
        gateway_config: GatewayConfig with log_file, log_max_bytes,
                        log_backup_count. Uses defaults if None.
    """
    log_path = _resolve_log_path(gateway_config)
    max_bytes = getattr(gateway_config, "log_max_bytes", None) or 10 * 1024 * 1024
    backup_count = getattr(gateway_config, "log_backup_count", None) or 5

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Single rotating handler for the entire process
    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)

    # Tune third-party log levels to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    logging.getLogger("slack_sdk").setLevel(logging.WARNING)

    # Redirect stdout/stderr so print() and unhandled tracebacks go to log
    sys.stdout = open(str(log_path), "a")
    sys.stderr = sys.stdout

    # Dump C stack trace on segfault
    faulthandler.enable(file=sys.stderr)


def _resolve_log_path(gateway_config=None):
    from pathlib import Path

    if gateway_config and getattr(gateway_config, "log_file", None):
        return Path(gateway_config.log_file)
    return get_config_dir() / "gateway.log"
