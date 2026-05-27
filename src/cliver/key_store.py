"""Encrypted key-value store backed by SQLite.

Stores named secrets (API keys, tokens) encrypted with Fernet.
The encryption key is derived from the machine's unique identifier,
so keys are automatically accessible on the same machine without
a password prompt.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import logging
import platform
import re
import socket
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KeyInfo:
    """Key metadata (no value)."""

    name: str
    description: str
    created_at: str
    updated_at: str


def _get_machine_id() -> str:
    """Get a platform-specific machine identifier."""
    system = platform.system()

    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                text=True,
                timeout=5,
            )
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
            if match:
                return match.group(1)
        except (subprocess.SubprocessError, OSError):
            pass

    elif system == "Linux":
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                mid = Path(path).read_text().strip()
                if mid:
                    return mid
            except OSError:
                continue

    elif system == "Windows":
        try:
            out = subprocess.check_output(
                ["reg", "query", r"HKLM\SOFTWARE\Microsoft\Cryptography", "/v", "MachineGuid"],
                text=True,
                timeout=5,
            )
            match = re.search(r"MachineGuid\s+REG_SZ\s+(\S+)", out)
            if match:
                return match.group(1)
        except (subprocess.SubprocessError, OSError):
            pass

    fallback = f"{socket.gethostname()}-{getpass.getuser()}"
    logger.debug("Using fallback machine ID: hostname + username")
    return hashlib.sha256(fallback.encode()).hexdigest()


class KeyStore:
    """Encrypted key-value store backed by SQLite."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._fernet = self._create_fernet()
        self._init_db()

    def _create_fernet(self):
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        machine_id = _get_machine_id()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"cliver-keystore-v1",
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
        return Fernet(key)

    def _init_db(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS keys ("
            "  name TEXT PRIMARY KEY,"
            "  encrypted_value BLOB NOT NULL,"
            "  description TEXT DEFAULT '',"
            "  created_at TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL"
            ")"
        )
        conn.commit()
        conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def set(self, name: str, value: str, description: str = "") -> None:
        encrypted = self._fernet.encrypt(value.encode())
        now = self._now()
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO keys (name, encrypted_value, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "encrypted_value=excluded.encrypted_value, "
            "description=excluded.description, "
            "updated_at=excluded.updated_at",
            (name, encrypted, description, now, now),
        )
        conn.commit()
        conn.close()

    def get(self, name: str) -> Optional[str]:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT encrypted_value FROM keys WHERE name=?", (name,)).fetchone()
        conn.close()
        if row is None:
            return None
        try:
            return self._fernet.decrypt(row[0]).decode()
        except Exception:
            logger.warning("Failed to decrypt key '%s'", name)
            return None

    def delete(self, name: str) -> bool:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("DELETE FROM keys WHERE name=?", (name,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def list_keys(self) -> List[KeyInfo]:
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT name, description, created_at, updated_at FROM keys ORDER BY name").fetchall()
        conn.close()
        return [KeyInfo(name=r[0], description=r[1], created_at=r[2], updated_at=r[3]) for r in rows]

    def has(self, name: str) -> bool:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT 1 FROM keys WHERE name=?", (name,)).fetchone()
        conn.close()
        return row is not None
