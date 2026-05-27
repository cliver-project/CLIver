"""
Template utilities for CLIver.

Provides Jinja2 template rendering for non-secret fields and a
three-layer secret resolution chain for secret fields:

1. KeyStore lookup (by name)
2. Environment variable (ALL_UPPERCASE names)
3. Literal value (as-is)

For non-secret fields, Jinja2 templates are still supported:
- {{ env.VARIABLE_NAME }}: Access any environment variable
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from jinja2 import BaseLoader, Environment

if TYPE_CHECKING:
    from cliver.key_store import KeyStore

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


# Global Jinja2 environment for non-secret template rendering
_jinja_env = Environment(loader=BaseLoader())
_jinja_env.globals["env"] = _EnvVarProxy()


def get_jinja_env() -> Environment:
    return _jinja_env


def render_template_if_needed(template_str: str, params: Dict[str, Any] = None) -> str:
    """Render a Jinja2 template string if it contains {{ }} markers.

    Used for non-secret fields (prompts, descriptions, etc.).
    For secret fields (api_key, token), use resolve_secret() instead.
    """
    if "{{" in template_str and "}}" in template_str:
        try:
            template = _jinja_env.from_string(template_str)
            return template.render(**(params or {}))
        except Exception as e:
            logger.warning(f"Failed to render template: {e}")
            return template_str
    return template_str


def resolve_secret(value: Optional[str], key_store: "KeyStore") -> Optional[str]:
    """Resolve a config value to its actual secret.

    Resolution order:
    1. KeyStore lookup -- if value matches a key name
    2. Environment variable -- if value is ALL_UPPERCASE (with underscores)
    3. Literal -- use as-is
    """
    if value is None:
        return None
    if not value:
        return value

    # 1. KeyStore
    resolved = key_store.get(value)
    if resolved is not None:
        return resolved

    # 2. Env var (ALL_UPPERCASE with underscores)
    stripped = value.replace("_", "")
    if stripped and value == value.upper() and stripped.isalpha():
        env_val = os.environ.get(value)
        if env_val:
            return env_val

    # 3. Literal
    return value
