"""Tests for KeyStore encrypted key-value store."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def key_store():
    from cliver.key_store import KeyStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "keys.db"
        yield KeyStore(db_path)


def test_set_and_get(key_store):
    key_store.set("openai_key", "sk-abc123")
    assert key_store.get("openai_key") == "sk-abc123"


def test_get_nonexistent(key_store):
    assert key_store.get("does_not_exist") is None


def test_set_overwrites(key_store):
    key_store.set("mykey", "value1")
    key_store.set("mykey", "value2")
    assert key_store.get("mykey") == "value2"


def test_set_with_description(key_store):
    key_store.set("mykey", "myval", description="My API key")
    keys = key_store.list_keys()
    assert len(keys) == 1
    assert keys[0].name == "mykey"
    assert keys[0].description == "My API key"


def test_delete_existing(key_store):
    key_store.set("mykey", "myval")
    assert key_store.delete("mykey") is True
    assert key_store.get("mykey") is None


def test_delete_nonexistent(key_store):
    assert key_store.delete("nope") is False


def test_list_keys_empty(key_store):
    assert key_store.list_keys() == []


def test_list_keys_multiple(key_store):
    key_store.set("key_a", "val_a", description="First key")
    key_store.set("key_b", "val_b", description="Second key")
    keys = key_store.list_keys()
    names = {k.name for k in keys}
    assert names == {"key_a", "key_b"}
    for k in keys:
        assert k.created_at
        assert k.updated_at


def test_has(key_store):
    assert key_store.has("mykey") is False
    key_store.set("mykey", "val")
    assert key_store.has("mykey") is True


def test_encryption_roundtrip(key_store):
    secret = "super-secret-api-key-with-special-chars-!@#$%^&*()"
    key_store.set("complex", secret)
    assert key_store.get("complex") == secret


def test_value_stored_encrypted():
    import sqlite3
    from cliver.key_store import KeyStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "keys.db"
        ks = KeyStore(db_path)
        ks.set("secret_key", "plaintext_value_12345")

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT encrypted_value FROM keys WHERE name='secret_key'").fetchone()
        conn.close()
        raw = row[0]
        assert b"plaintext_value_12345" not in raw


def test_get_machine_id_returns_string():
    from cliver.key_store import _get_machine_id

    mid = _get_machine_id()
    assert isinstance(mid, str)
    assert len(mid) > 0


def test_different_db_paths_work():
    from cliver.key_store import KeyStore

    with tempfile.TemporaryDirectory() as tmpdir:
        ks1 = KeyStore(Path(tmpdir) / "a.db")
        ks2 = KeyStore(Path(tmpdir) / "b.db")
        ks1.set("key1", "val1")
        assert ks2.get("key1") is None
        ks2.set("key2", "val2")
        assert ks1.get("key2") is None
