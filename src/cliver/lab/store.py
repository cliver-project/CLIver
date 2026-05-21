"""LabStore — SQLite persistence for AI Labs and golden tests."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from cliver.db import SQLiteStore
from cliver.lab.models import GoldenTest, Lab

_SCHEMA = """
CREATE TABLE IF NOT EXISTS labs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS golden_tests (
    id TEXT PRIMARY KEY,
    lab_id TEXT NOT NULL REFERENCES labs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    input TEXT NOT NULL,
    expected_output TEXT NOT NULL,
    expected_files TEXT DEFAULT '[]',
    sort_order INTEGER DEFAULT 0
);
"""


class LabStore:
    """CRUD store for AI Labs and their golden tests."""

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._store: Optional[SQLiteStore] = None

    def _get_store(self) -> SQLiteStore:
        if self._store is None:
            self._store = SQLiteStore(self._db_path)
            self._store.execute_schema(_SCHEMA)
        return self._store

    # -- Labs ---------------------------------------------------------------

    def create_lab(self, title: str, description: str = "") -> Lab:
        lab = Lab(title=title, description=description)
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO labs (id, title, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (lab.id, lab.title, lab.description, lab.created_at, lab.updated_at),
            )
        return lab

    def list_labs(self) -> List[Lab]:
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, title, description, created_at, updated_at FROM labs ORDER BY updated_at DESC"
            ).fetchall()
        return [Lab(**dict(r)) for r in rows]

    def get_lab(self, lab_id: str) -> Optional[Lab]:
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, title, description, created_at, updated_at FROM labs WHERE id = ?",
                (lab_id,),
            ).fetchone()
        if row is None:
            return None
        return Lab(**dict(row))

    def update_lab(self, lab_id: str, title: str = "", description: str = "") -> Optional[Lab]:
        existing = self.get_lab(lab_id)
        if existing is None:
            return None
        if title:
            existing.title = title
        if description:
            existing.description = description
        from cliver.lab.models import _now

        existing.updated_at = _now()
        with self._get_store().write() as db:
            db.execute(
                "UPDATE labs SET title = ?, description = ?, updated_at = ? WHERE id = ?",
                (existing.title, existing.description, existing.updated_at, lab_id),
            )
        return existing

    def delete_lab(self, lab_id: str) -> bool:
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM labs WHERE id = ?", (lab_id,))
        return cursor.rowcount > 0

    # -- Golden Tests -------------------------------------------------------

    def list_golden_tests(self, lab_id: str) -> List[GoldenTest]:
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, lab_id, name, input, expected_output, expected_files, sort_order "
                "FROM golden_tests WHERE lab_id = ? ORDER BY sort_order",
                (lab_id,),
            ).fetchall()
        return [GoldenTest(**dict(r)) for r in rows]

    def create_golden_test(
        self,
        lab_id: str,
        name: str,
        input: str,
        expected_output: str,
        expected_files: str = "[]",
    ) -> GoldenTest:
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM golden_tests WHERE lab_id = ?",
                (lab_id,),
            ).fetchone()
        next_order = row[0] if row else 0

        gt = GoldenTest(
            lab_id=lab_id,
            name=name,
            input=input,
            expected_output=expected_output,
            expected_files=expected_files,
            sort_order=next_order,
        )
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO golden_tests (id, lab_id, name, input, expected_output, expected_files, sort_order) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (gt.id, gt.lab_id, gt.name, gt.input, gt.expected_output, gt.expected_files, gt.sort_order),
            )
        return gt

    def update_golden_test(
        self,
        test_id: str,
        name: Optional[str] = None,
        input: Optional[str] = None,
        expected_output: Optional[str] = None,
        expected_files: Optional[str] = None,
    ) -> Optional[GoldenTest]:
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, lab_id, name, input, expected_output, expected_files, sort_order "
                "FROM golden_tests WHERE id = ?",
                (test_id,),
            ).fetchone()
        if row is None:
            return None
        gt = GoldenTest(**dict(row))
        if name is not None:
            gt.name = name
        if input is not None:
            gt.input = input
        if expected_output is not None:
            gt.expected_output = expected_output
        if expected_files is not None:
            gt.expected_files = expected_files
        with self._get_store().write() as db:
            db.execute(
                "UPDATE golden_tests SET name = ?, input = ?, expected_output = ?, expected_files = ? WHERE id = ?",
                (gt.name, gt.input, gt.expected_output, gt.expected_files, test_id),
            )
        return gt

    def delete_golden_test(self, test_id: str) -> bool:
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM golden_tests WHERE id = ?", (test_id,))
        return cursor.rowcount > 0

    def close(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
