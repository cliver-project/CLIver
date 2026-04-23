"""SQLiteStore — per-file write-serialized SQLite access.

Each database file gets one SQLiteStore instance. Writes are serialized
via a threading.Lock. Reads go through freely (WAL mode handles concurrent
readers). Per-thread connections via threading.local() ensure thread safety.

Usage:
    store = get_store(Path("my.db"))
    with store.write() as conn:
        conn.execute("INSERT ...")
    with store.read() as conn:
        conn.execute("SELECT ...")
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict


class SQLiteStore:
    """Per-file SQLite store with write serialization."""

    def __init__(self, db_path: Path, busy_timeout_ms: int = 5000):
        self._db_path = Path(db_path)
        self._busy_timeout_ms = busy_timeout_ms
        self._write_lock = threading.Lock()
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self._db_path),
                timeout=self._busy_timeout_ms / 1000,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def read(self):
        yield self._get_conn()

    @contextmanager
    def write(self):
        with self._write_lock:
            conn = self._get_conn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute_schema(self, schema: str):
        with self._write_lock:
            conn = self._get_conn()
            conn.executescript(schema)

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


_stores: Dict[Path, SQLiteStore] = {}
_stores_lock = threading.Lock()


def get_store(db_path: Path, busy_timeout_ms: int = 5000) -> SQLiteStore:
    resolved = db_path.resolve()
    with _stores_lock:
        if resolved not in _stores:
            _stores[resolved] = SQLiteStore(resolved, busy_timeout_ms)
        return _stores[resolved]
