"""Tests for Project, Issue, and Scenario models."""


def test_project_defaults():
    from cliver.project.models import Project

    p = Project(id="proj_abc123", name="Research Q2")
    assert p.id == "proj_abc123"
    assert p.name == "Research Q2"
    assert p.description == ""
    assert p.source == "local"
    assert p.source_url is None
    assert p.created_at


def test_project_with_metadata():
    from cliver.project.models import Project

    p = Project(
        id="proj_xyz",
        name="My Project",
        description="A test project",
        source="github",
        source_url="https://github.com/user/repo",
    )
    assert p.source == "github"
    assert p.source_url == "https://github.com/user/repo"


def test_issue_defaults():
    from cliver.project.models import Issue

    i = Issue(id="iss_abc123", project_id="proj_1", title="Fix bug")
    assert i.status == "open"
    assert i.priority == "medium"
    assert i.labels == []
    assert i.assigned_agent is None
    assert i.scenario_id is None
    assert i.lab_id is None


def test_issue_full():
    from cliver.project.models import Issue

    i = Issue(
        id="iss_xyz",
        project_id="proj_1",
        title="Research transformers",
        description="Survey recent papers",
        status="in_progress",
        priority="high",
        labels=["research", "ai"],
        assigned_agent="researcher",
        scenario_id="research-ai-lab",
        lab_id="lab_abc123",
    )
    assert i.status == "in_progress"
    assert i.priority == "high"
    assert i.labels == ["research", "ai"]
    assert i.assigned_agent == "researcher"
    assert i.scenario_id == "research-ai-lab"
    assert i.lab_id == "lab_abc123"


def test_valid_statuses():
    from cliver.project.models import VALID_STATUSES

    assert "open" in VALID_STATUSES
    assert "in_progress" in VALID_STATUSES
    assert "completed" in VALID_STATUSES
    assert "closed" in VALID_STATUSES


def test_valid_priorities():
    from cliver.project.models import VALID_PRIORITIES

    assert "low" in VALID_PRIORITIES
    assert "medium" in VALID_PRIORITIES
    assert "high" in VALID_PRIORITIES
    assert "critical" in VALID_PRIORITIES


def test_scenario_defaults():
    from cliver.project.models import Scenario

    s = Scenario(id="research-ai-lab", name="research-ai-lab", display_name="Research AI Lab")
    assert s.description == ""
    assert s.tags == []
    assert s.agent_requirements == []
    assert s.source == "builtin"
    assert s.path is None


def test_scenario_full():
    from cliver.project.models import Scenario

    s = Scenario(
        id="research-ai-lab",
        name="research-ai-lab",
        display_name="Research AI Lab",
        description="Paper analysis",
        tags=["research", "papers"],
        agent_requirements=["cliver"],
        source="user",
        path="/home/user/.cliver/scenarios/research-ai-lab",
    )
    assert s.tags == ["research", "papers"]
    assert s.agent_requirements == ["cliver"]
    assert s.source == "user"


def test_issue_serialization():
    from cliver.project.models import Issue

    i = Issue(id="iss_1", project_id="proj_1", title="Test", labels=["a", "b"])
    d = i.model_dump()
    assert d["labels"] == ["a", "b"]
    assert d["status"] == "open"

    i2 = Issue.model_validate(d)
    assert i2.labels == ["a", "b"]
