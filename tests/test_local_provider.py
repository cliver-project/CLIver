"""Tests for LocalProvider SQLite implementation."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def provider():
    from cliver.project.local_provider import LocalProvider

    with tempfile.TemporaryDirectory() as tmpdir:
        yield LocalProvider(Path(tmpdir) / "projects.db")


# --- Project Tests ---

@pytest.mark.asyncio
async def test_create_project(provider):
    p = await provider.create_project("Research Q2", "Papers survey")
    assert p.id.startswith("proj_")
    assert p.name == "Research Q2"
    assert p.description == "Papers survey"
    assert p.source == "local"


@pytest.mark.asyncio
async def test_get_project(provider):
    created = await provider.create_project("Test")
    fetched = await provider.get_project(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "Test"


@pytest.mark.asyncio
async def test_get_project_not_found(provider):
    assert await provider.get_project("proj_nonexistent") is None


@pytest.mark.asyncio
async def test_list_projects(provider):
    await provider.create_project("First")
    await provider.create_project("Second")
    projects = await provider.list_projects()
    assert len(projects) == 2


@pytest.mark.asyncio
async def test_update_project(provider):
    p = await provider.create_project("Original")
    p.name = "Updated"
    p.description = "New desc"
    await provider.update_project(p)
    fetched = await provider.get_project(p.id)
    assert fetched.name == "Updated"
    assert fetched.description == "New desc"


@pytest.mark.asyncio
async def test_delete_project(provider):
    p = await provider.create_project("To Delete")
    assert await provider.delete_project(p.id) is True
    assert await provider.get_project(p.id) is None
    assert await provider.delete_project(p.id) is False


@pytest.mark.asyncio
async def test_delete_project_cascades_issues(provider):
    p = await provider.create_project("Parent")
    await provider.create_issue(p.id, "Child Issue")
    await provider.delete_project(p.id)
    issues = await provider.list_issues(p.id)
    assert len(issues) == 0


# --- Issue Tests ---

@pytest.mark.asyncio
async def test_create_issue(provider):
    p = await provider.create_project("Proj")
    i = await provider.create_issue(p.id, "Fix bug", description="It's broken", priority="high")
    assert i.id.startswith("iss_")
    assert i.title == "Fix bug"
    assert i.status == "open"
    assert i.priority == "high"


@pytest.mark.asyncio
async def test_create_issue_with_labels(provider):
    p = await provider.create_project("Proj")
    i = await provider.create_issue(p.id, "Task", labels=["research", "ai"])
    assert i.labels == ["research", "ai"]
    fetched = await provider.get_issue(i.id)
    assert fetched.labels == ["research", "ai"]


@pytest.mark.asyncio
async def test_create_issue_with_agent(provider):
    p = await provider.create_project("Proj")
    i = await provider.create_issue(p.id, "Task", assigned_agent="researcher")
    assert i.assigned_agent == "researcher"


@pytest.mark.asyncio
async def test_get_issue(provider):
    p = await provider.create_project("Proj")
    created = await provider.create_issue(p.id, "Test Issue")
    fetched = await provider.get_issue(created.id)
    assert fetched is not None
    assert fetched.title == "Test Issue"


@pytest.mark.asyncio
async def test_list_issues(provider):
    p = await provider.create_project("Proj")
    await provider.create_issue(p.id, "Issue 1")
    await provider.create_issue(p.id, "Issue 2")
    issues = await provider.list_issues(p.id)
    assert len(issues) == 2


@pytest.mark.asyncio
async def test_list_issues_filter_status(provider):
    p = await provider.create_project("Proj")
    i1 = await provider.create_issue(p.id, "Open")
    i2 = await provider.create_issue(p.id, "Done")
    i2.status = "completed"
    await provider.update_issue(i2)
    open_issues = await provider.list_issues(p.id, status="open")
    assert len(open_issues) == 1
    assert open_issues[0].title == "Open"


@pytest.mark.asyncio
async def test_list_issues_filter_labels(provider):
    p = await provider.create_project("Proj")
    await provider.create_issue(p.id, "AI Task", labels=["ai"])
    await provider.create_issue(p.id, "Web Task", labels=["web"])
    ai_issues = await provider.list_issues(p.id, labels=["ai"])
    assert len(ai_issues) == 1
    assert ai_issues[0].title == "AI Task"


@pytest.mark.asyncio
async def test_update_issue(provider):
    p = await provider.create_project("Proj")
    i = await provider.create_issue(p.id, "Original")
    i.title = "Updated"
    i.status = "in_progress"
    i.notebook_id = "nb_abc123"
    await provider.update_issue(i)
    fetched = await provider.get_issue(i.id)
    assert fetched.title == "Updated"
    assert fetched.status == "in_progress"
    assert fetched.notebook_id == "nb_abc123"


@pytest.mark.asyncio
async def test_delete_issue(provider):
    p = await provider.create_project("Proj")
    i = await provider.create_issue(p.id, "To Delete")
    assert await provider.delete_issue(i.id) is True
    assert await provider.get_issue(i.id) is None
    assert await provider.delete_issue(i.id) is False
