"""Tests for SQLiteStore — per-file write-serialized SQLite access."""

import threading

import pytest

from cliver.db import SQLiteStore, get_store


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "test.db")


@pytest.fixture
def store_with_table(store):
    store.execute_schema("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, value TEXT)")
    return store


class TestReadWrite:
    def test_write_and_read(self, store_with_table):
        with store_with_table.write() as conn:
            conn.execute("INSERT INTO items (value) VALUES (?)", ("hello",))

        with store_with_table.read() as conn:
            row = conn.execute("SELECT value FROM items WHERE id = 1").fetchone()
            assert row["value"] == "hello"

    def test_write_rollback_on_error(self, store_with_table):
        try:
            with store_with_table.write() as conn:
                conn.execute("INSERT INTO items (value) VALUES (?)", ("bad",))
                raise ValueError("forced error")
        except ValueError:
            pass

        with store_with_table.read() as conn:
            row = conn.execute("SELECT COUNT(*) FROM items").fetchone()
            assert row[0] == 0

    def test_read_does_not_block_read(self, store_with_table):
        with store_with_table.write() as conn:
            conn.execute("INSERT INTO items (value) VALUES (?)", ("a",))

        with store_with_table.read() as c1:
            with store_with_table.read() as c2:
                r1 = c1.execute("SELECT COUNT(*) FROM items").fetchone()[0]
                r2 = c2.execute("SELECT COUNT(*) FROM items").fetchone()[0]
                assert r1 == r2 == 1


class TestSchema:
    def test_execute_schema(self, store):
        store.execute_schema("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)")
        with store.write() as conn:
            conn.execute("INSERT INTO t (id) VALUES (1)")
        with store.read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


class TestRegistry:
    def test_get_store_returns_same_instance(self, tmp_path):
        db_path = tmp_path / "shared.db"
        s1 = get_store(db_path)
        s2 = get_store(db_path)
        assert s1 is s2

    def test_get_store_different_paths(self, tmp_path):
        s1 = get_store(tmp_path / "a.db")
        s2 = get_store(tmp_path / "b.db")
        assert s1 is not s2


class TestClose:
    def test_close_and_reopen(self, store_with_table):
        with store_with_table.write() as conn:
            conn.execute("INSERT INTO items (value) VALUES (?)", ("persist",))
        store_with_table.close()

        with store_with_table.read() as conn:
            row = conn.execute("SELECT value FROM items WHERE id = 1").fetchone()
            assert row["value"] == "persist"


class TestWriteSerialization:
    def test_concurrent_writes_serialize(self, store_with_table):
        results = []

        def writer(val):
            with store_with_table.write() as conn:
                conn.execute("INSERT INTO items (value) VALUES (?)", (val,))
                results.append(val)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        with store_with_table.read() as conn:
            count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            assert count == 10
