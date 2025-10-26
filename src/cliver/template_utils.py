"""
Template utilities for CLIver.

This module provides Jinja2 template rendering functionality with performance optimization.
"""

import logging
import os
from typing import Any, Dict

from jinja2 import Environment, BaseLoader

logger = logging.getLogger(__name__)


def _get_cliver_env_vars():
    """Get environment variables with CLIVER_ prefix as a dictionary."""
    cliver_env = {}
    for key, value in os.environ.items():
        if key.startswith('CLIVER_'):
            # Remove the CLIVER_ prefix and convert to lowercase for consistency
            cliver_key = key[7:].lower()
            cliver_env[cliver_key] = value
    return cliver_env


def _get_secret_from_keyring(secret_name: str) -> str:
    """
    Get a secret from the keyring.

    Args:
        secret_name: The name to retrieve the secret for

    Returns:
        The secret value

    Note:
        Requires the 'keyring' package to be installed:
        pip install keyring
    """
    try:
        import keyring
        return keyring.get_password("cliver", secret_name) or ""
    except ImportError:
        logger.warning("Keyring package not installed. Please install it with: pip install keyring")
        return ""
    except Exception as e:
        logger.warning(f"Failed to get secret from keyring: {e}")
        return ""


# Global Jinja2 environment for template rendering
_jinja_env = Environment(loader=BaseLoader())
_jinja_env.globals['env'] = _get_cliver_env_vars()
_jinja_env.globals['keyring'] = _get_secret_from_keyring

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
    if '{{' in template_str and '}}' in template_str:
        try:
            template = _jinja_env.from_string(template_str)
            return template.render(**(params or {}))
        except Exception as e:
            logger.warning(f"Failed to render template: {e}")
            return template_str
    return template_str