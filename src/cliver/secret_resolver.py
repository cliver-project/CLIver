"""
Secret resolver for CLIver.

Resolves secret references to actual values. Supports:
- Plain text: used as-is (e.g., "sk-abc123")
- Vault reference: "vault:<service-name>:<key-name>" reads from system keyring/keychain
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

VAULT_PREFIX = "vault:"


def resolve_secret(value: Optional[str]) -> Optional[str]:
    """
    Resolve a secret value that may be a vault reference.

    Args:
        value: The raw value from config. Can be:
            - None: returns None
            - "vault:<service>:<key>": reads from system keyring/keychain
            - Any other string: returned as-is (plain text)

    Returns:
        The resolved secret value, or None if not found
    """
    if value is None:
        return None

    if not value.startswith(VAULT_PREFIX):
        return value

    # Parse vault:<service-name>:<key-name>
    parts = value[len(VAULT_PREFIX) :].split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        logger.error(f"Invalid vault reference format: '{value}'. Expected 'vault:<service-name>:<key-name>'")
        return None

    service_name, key_name = parts
    return _read_from_keyring(service_name, key_name)


def _read_from_keyring(service_name: str, key_name: str) -> Optional[str]:
    """
    Read a secret from the system keyring/keychain.

    On macOS this uses Keychain, on Linux it uses the system keyring
    (e.g., GNOME Keyring, KWallet), on Windows it uses Credential Locker.

    Args:
        service_name: The service/application name in the keyring
        key_name: The key/account name to look up

    Returns:
        The secret value, or None if not found
    """
    try:
        import keyring

        secret = keyring.get_password(service_name, key_name)
        if secret is None:
            logger.warning(
                f"No secret found in keyring for service='{service_name}', key='{key_name}'. "
                f'Set it with: python -c "import keyring; '
                f"keyring.set_password('{service_name}', '{key_name}', 'your-secret')\""
            )
        return secret
    except ImportError:
        logger.error("The 'keyring' package is required for vault references. Install it with: pip install keyring")
        return None
    except Exception as e:
        logger.error(f"Failed to read from keyring (service='{service_name}', key='{key_name}'): {e}")
        return None
