"""LocalProvider — SQLite-backed project/issue storage."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from cliver.db import SQLiteStore
from cliver.project.models import Issue, Project
from cliver.project.provider import ProjectProvider

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    source TEXT DEFAULT 'local',
    source_url TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS issues (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT DEFAULT 'open',
    priority TEXT DEFAULT 'medium',
    labels TEXT DEFAULT '[]',
    assigned_agent TEXT,
    scenario_id TEXT,
    lab_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class LocalProvider(ProjectProvider):
    """SQLite-backed local storage for projects and issues."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._store = SQLiteStore(db_path)
        self._store.execute_schema(_SCHEMA)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _gen_project_id(self) -> str:
        return f"proj_{uuid.uuid4().hex[:8]}"

    def _gen_issue_id(self) -> str:
        return f"iss_{uuid.uuid4().hex[:8]}"

    # --- Project CRUD ---

    async def create_project(self, name: str, description: str = "") -> Project:
        now = self._now()
        project = Project(
            id=self._gen_project_id(),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )
        with self._store.write() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, source, source_url, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    project.id,
                    project.name,
                    project.description,
                    project.source,
                    project.source_url,
                    project.created_at,
                    project.updated_at,
                ),
            )
        return project

    async def get_project(self, project_id: str) -> Optional[Project]:
        with self._store.read() as conn:
            row = conn.execute(
                "SELECT id, name, description, source, source_url, created_at, updated_at FROM projects WHERE id=?",
                (project_id,),
            ).fetchone()
        if not row:
            return None
        return Project(
            id=row[0],
            name=row[1],
            description=row[2],
            source=row[3],
            source_url=row[4],
            created_at=row[5],
            updated_at=row[6],
        )

    async def list_projects(self) -> List[Project]:
        with self._store.read() as conn:
            rows = conn.execute(
                "SELECT id, name, description, source, source_url, created_at, updated_at "
                "FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [
            Project(
                id=r[0],
                name=r[1],
                description=r[2],
                source=r[3],
                source_url=r[4],
                created_at=r[5],
                updated_at=r[6],
            )
            for r in rows
        ]

    async def update_project(self, project: Project) -> None:
        project.updated_at = self._now()
        with self._store.write() as conn:
            conn.execute(
                "UPDATE projects SET name=?, description=?, source=?, source_url=?, updated_at=? WHERE id=?",
                (project.name, project.description, project.source, project.source_url, project.updated_at, project.id),
            )

    async def delete_project(self, project_id: str) -> bool:
        with self._store.write() as conn:
            conn.execute("DELETE FROM issues WHERE project_id=?", (project_id,))
            cursor = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            return cursor.rowcount > 0

    # --- Issue CRUD ---

    async def create_issue(
        self,
        project_id: str,
        title: str,
        description: str = "",
        priority: str = "medium",
        labels: Optional[List[str]] = None,
        assigned_agent: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> Issue:
        now = self._now()
        issue = Issue(
            id=self._gen_issue_id(),
            project_id=project_id,
            title=title,
            description=description,
            priority=priority,
            labels=labels or [],
            assigned_agent=assigned_agent,
            scenario_id=scenario_id,
            created_at=now,
            updated_at=now,
        )
        with self._store.write() as conn:
            conn.execute(
                "INSERT INTO issues (id, project_id, title, description, status, priority, "
                "labels, assigned_agent, scenario_id, lab_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    issue.id,
                    issue.project_id,
                    issue.title,
                    issue.description,
                    issue.status,
                    issue.priority,
                    json.dumps(issue.labels),
                    issue.assigned_agent,
                    issue.scenario_id,
                    issue.lab_id,
                    issue.created_at,
                    issue.updated_at,
                ),
            )
        return issue

    async def get_issue(self, issue_id: str) -> Optional[Issue]:
        with self._store.read() as conn:
            row = conn.execute(
                "SELECT id, project_id, title, description, status, priority, labels, "
                "assigned_agent, scenario_id, lab_id, created_at, updated_at "
                "FROM issues WHERE id=?",
                (issue_id,),
            ).fetchone()
        if not row:
            return None
        return Issue(
            id=row[0],
            project_id=row[1],
            title=row[2],
            description=row[3],
            status=row[4],
            priority=row[5],
            labels=json.loads(row[6] or "[]"),
            assigned_agent=row[7],
            scenario_id=row[8],
            lab_id=row[9],
            created_at=row[10],
            updated_at=row[11],
        )

    async def list_issues(
        self,
        project_id: str,
        status: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> List[Issue]:
        with self._store.read() as conn:
            query = (
                "SELECT id, project_id, title, description, status, priority, labels, "
                "assigned_agent, scenario_id, lab_id, created_at, updated_at "
                "FROM issues WHERE project_id=?"
            )
            params: list = [project_id]
            if status:
                query += " AND status=?"
                params.append(status)
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query, params).fetchall()

        issues = [
            Issue(
                id=r[0],
                project_id=r[1],
                title=r[2],
                description=r[3],
                status=r[4],
                priority=r[5],
                labels=json.loads(r[6] or "[]"),
                assigned_agent=r[7],
                scenario_id=r[8],
                lab_id=r[9],
                created_at=r[10],
                updated_at=r[11],
            )
            for r in rows
        ]

        if labels:
            label_set = set(labels)
            issues = [i for i in issues if label_set.intersection(i.labels)]

        return issues

    async def update_issue(self, issue: Issue) -> None:
        issue.updated_at = self._now()
        with self._store.write() as conn:
            conn.execute(
                "UPDATE issues SET title=?, description=?, status=?, priority=?, labels=?, "
                "assigned_agent=?, scenario_id=?, lab_id=?, updated_at=? WHERE id=?",
                (
                    issue.title,
                    issue.description,
                    issue.status,
                    issue.priority,
                    json.dumps(issue.labels),
                    issue.assigned_agent,
                    issue.scenario_id,
                    issue.lab_id,
                    issue.updated_at,
                    issue.id,
                ),
            )

    async def delete_issue(self, issue_id: str) -> bool:
        with self._store.write() as conn:
            cursor = conn.execute("DELETE FROM issues WHERE id=?", (issue_id,))
            return cursor.rowcount > 0
