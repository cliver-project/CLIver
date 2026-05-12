"""ProjectProvider ABC — abstract interface for project/issue storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from cliver.project.models import Issue, Project


class ProjectProvider(ABC):
    """Abstract interface for project/issue storage.

    Implementations: LocalProvider (SQLite), future GitHubProvider, JiraProvider.
    """

    @abstractmethod
    async def create_project(self, name: str, description: str = "") -> Project: ...

    @abstractmethod
    async def get_project(self, project_id: str) -> Optional[Project]: ...

    @abstractmethod
    async def list_projects(self) -> List[Project]: ...

    @abstractmethod
    async def update_project(self, project: Project) -> None: ...

    @abstractmethod
    async def delete_project(self, project_id: str) -> bool: ...

    @abstractmethod
    async def create_issue(
        self,
        project_id: str,
        title: str,
        description: str = "",
        priority: str = "medium",
        labels: Optional[List[str]] = None,
        assigned_agent: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> Issue: ...

    @abstractmethod
    async def get_issue(self, issue_id: str) -> Optional[Issue]: ...

    @abstractmethod
    async def list_issues(
        self,
        project_id: str,
        status: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> List[Issue]: ...

    @abstractmethod
    async def update_issue(self, issue: Issue) -> None: ...

    @abstractmethod
    async def delete_issue(self, issue_id: str) -> bool: ...
