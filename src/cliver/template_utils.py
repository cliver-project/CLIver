"""
Template utilities for CLIver.

This module provides Jinja2 template rendering functionality with performance optimization.
Templates are used in config files, prompts, and workflows. Supported template functions:

- {{ env.VARIABLE_NAME }}: Access any environment variable
- {{ keyring('service', 'key') }}: Read a secret from the system keyring
"""

import logging
import os
from typing import Any, Dict, Optional

from jinja2 import BaseLoader, Environment

logger = logging.getLogger(__name__)


class _EnvVarProxy:
    """Dict-like proxy over os.environ that resolves env vars at access time."""

    def __getattr__(self, name: str) -> str:
        return os.environ.get(name, "")

    def __getitem__(self, name: str) -> str:
        return os.environ.get(name, "")

    def __contains__(self, name: str) -> bool:
        return name in os.environ

    def __repr__(self) -> str:
        return f"EnvVarProxy({dict(os.environ)})"


def _get_secret_from_keyring(service: str, key: Optional[str] = None) -> str:
    """
    Get a secret from the system keyring.

    Args:
        service: The service/application name in the keyring
        key: The key/account name to look up

    Returns:
        The secret value, or empty string if not found

    Usage in templates:
        {{ keyring('cliver', 'api_key') }}
    """
    if key is None:
        # Backward compat: single arg treated as key with 'cliver' service
        key = service
        service = "cliver"

    try:
        import keyring

        secret = keyring.get_password(service, key)
        if secret is None:
            logger.warning(
                f"No secret found in keyring for service='{service}', key='{key}'. "
                f'Set it with: python -c "import keyring; '
                f"keyring.set_password('{service}', '{key}', 'your-secret')\""
            )
            return ""
        return secret
    except ImportError:
        logger.warning("Keyring package not installed. Please install it with: pip install keyring")
        return ""
    except Exception as e:
        logger.warning(f"Failed to get secret from keyring: {e}")
        return ""


# Global Jinja2 environment for template rendering
_jinja_env = Environment(loader=BaseLoader())
_jinja_env.globals["env"] = _EnvVarProxy()
_jinja_env.globals["keyring"] = _get_secret_from_keyring


def get_jinja_env() -> Environment:
    return _jinja_env


# more global contexts can be added here.


def render_template_if_needed(template_str: str, params: Dict[str, Any] = None) -> str:
    """
    Render a template string if it contains Jinja2 template markers.

    This function checks if the template string contains '{{' and '}}' markers
    and only renders it through Jinja2 if those markers are present for performance.

    Args:
        template_str: The template string to potentially render
        params: Parameters to use for template rendering

    Returns:
        The rendered template string or the original string if no rendering was needed
    """
    # Check if template contains markers for performance optimization
    if "{{" in template_str and "}}" in template_str:
        try:
            template = _jinja_env.from_string(template_str)
            return template.render(**(params or {}))
        except Exception as e:
            logger.warning(f"Failed to render template: {e}")
            return template_str
    return template_str
