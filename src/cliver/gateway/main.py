"""Gateway subprocess entry point.

Invoked by `python -m cliver.gateway.main` from the CLI's _start_gateway().
Runs in a fresh process (no fork) so macOS mach ports, DNS resolver,
and Keychain all work correctly.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def main():
    from cliver.config import ConfigManager
    from cliver.gateway.gateway import Gateway
    from cliver.gateway.logging_config import configure_gateway_logging
    from cliver.util import get_config_dir

    config_dir = get_config_dir()
    config_manager = ConfigManager(config_dir)
    cfg = config_manager.config

    configure_gateway_logging(cfg.gateway)

    gw = Gateway(config_dir=config_dir, resolved_config=cfg)
    try:
        gw.init()
        logger.info("Gateway initialized (default model: %s)", gw._get_default_model_name())
    except Exception as e:
        logger.error("AgentCore init failed: %s", e)

    try:
        gw.run()
    except SystemExit as e:
        logger.info("Gateway exited with code %s", e.code)
    except Exception:
        logger.exception("Gateway crashed")
        sys.exit(1)


if __name__ == "__main__":
    main()
