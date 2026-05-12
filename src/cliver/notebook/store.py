"""NotebookStore — CRUD for notebooks with JSON files + SQLite index."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.notebook.models import Cell, Notebook, NotebookSummary

logger = logging.getLogger(__name__)


class NotebookStore:
    """CRUD for notebooks. JSON files are source of truth, SQLite is index."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._notebooks_dir = data_dir / "notebooks"
        self._db_path = data_dir / "notebooks.db"
        self._notebooks_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS notebooks ("
            "  id TEXT PRIMARY KEY,"
            "  title TEXT NOT NULL,"
            "  description TEXT DEFAULT '',"
            "  scenario_id TEXT,"
            "  cell_count INTEGER DEFAULT 0,"
            "  status TEXT DEFAULT 'idle',"
            "  created_at TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL"
            ")"
        )
        conn.commit()
        conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _gen_id(self) -> str:
        return f"nb_{uuid.uuid4().hex[:8]}"

    def create(
        self,
        title: str,
        description: str = "",
        scenario_id: Optional[str] = None,
        default_agent: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cells: Optional[list] = None,
    ) -> Notebook:
        now = self._now()
        nb_id = self._gen_id()

        cell_objects = []
        if cells:
            for c in cells:
                if isinstance(c, dict):
                    cell_objects.append(Cell(**c))
                elif isinstance(c, Cell):
                    cell_objects.append(c)

        nb = Notebook(
            id=nb_id,
            title=title,
            description=description,
            scenario_id=scenario_id,
            default_agent=default_agent,
            context=context or {},
            created_at=now,
            updated_at=now,
            cells=cell_objects,
        )

        self._write_json(nb)
        self._upsert_index(nb)
        return nb

    def get(self, notebook_id: str) -> Optional[Notebook]:
        path = self._notebooks_dir / f"{notebook_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Notebook.model_validate(data)
        except Exception as e:
            logger.warning("Failed to load notebook %s: %s", notebook_id, e)
            return None

    def list_all(self) -> List[NotebookSummary]:
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            "SELECT id, title, description, scenario_id, cell_count, status, "
            "created_at, updated_at FROM notebooks ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [
            NotebookSummary(
                id=r[0],
                title=r[1],
                description=r[2] or "",
                scenario_id=r[3],
                cell_count=r[4],
                status=r[5],
                created_at=r[6],
                updated_at=r[7],
            )
            for r in rows
        ]

    def update(self, notebook: Notebook) -> None:
        notebook.updated_at = self._now()
        self._write_json(notebook)
        self._upsert_index(notebook)

    def delete(self, notebook_id: str) -> bool:
        path = self._notebooks_dir / f"{notebook_id}.json"
        if not path.exists():
            return False
        path.unlink()
        conn = sqlite3.connect(self._db_path)
        conn.execute("DELETE FROM notebooks WHERE id=?", (notebook_id,))
        conn.commit()
        conn.close()
        return True

    def save_cell_output(
        self,
        notebook_id: str,
        cell_id: str,
        outputs: Dict[str, Any],
        status: str,
        error: Optional[str] = None,
        duration_ms: int = 0,
    ) -> None:
        nb = self.get(notebook_id)
        if not nb:
            raise ValueError(f"Notebook '{notebook_id}' not found")
        cell = nb.get_cell(cell_id)
        if not cell:
            raise ValueError(f"Cell '{cell_id}' not found in notebook '{notebook_id}'")

        cell.outputs = outputs
        cell.status = status
        cell.error = error
        cell.duration_ms = duration_ms
        self.update(nb)

    def _write_json(self, notebook: Notebook) -> None:
        path = self._notebooks_dir / f"{notebook.id}.json"
        tmp_path = path.with_suffix(".tmp")
        data = notebook.model_dump()
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.rename(tmp_path, path)

    def _upsert_index(self, notebook: Notebook) -> None:
        status = "idle"
        for c in notebook.cells:
            if c.status == "error":
                status = "error"
                break
            if c.status == "completed":
                status = "completed"

        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO notebooks (id, title, description, scenario_id, cell_count, "
            "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, "
            "description=excluded.description, scenario_id=excluded.scenario_id, "
            "cell_count=excluded.cell_count, status=excluded.status, "
            "updated_at=excluded.updated_at",
            (
                notebook.id,
                notebook.title,
                notebook.description,
                notebook.scenario_id,
                len(notebook.cells),
                status,
                notebook.created_at,
                notebook.updated_at,
            ),
        )
        conn.commit()
        conn.close()
