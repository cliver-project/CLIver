"""Tests for ScenarioRegistry."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from cliver.project.models import Issue


@pytest.fixture
def scenario_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "scenarios" / "test-scenario"
        d.mkdir(parents=True)

        meta = {
            "name": "test-scenario",
            "display_name": "Test Scenario",
            "description": "A test scenario",
            "tags": ["test", "demo"],
            "agent_requirements": ["cliver"],
        }
        (d / "scenario.yaml").write_text(yaml.dump(meta))

        template = {
            "$schema": "cliver-lab-v1",
            "title": "${issue.title}",
            "description": "${issue.description}",
            "default_agent": "cliver",
            "cells": [
                {
                    "id": "setup",
                    "type": "config",
                    "title": "Setup",
                    "inputs": {"schema": {"domain": {"type": "text", "default": "${issue.title}"}}},
                    "outputs": {},
                    "status": "idle",
                },
                {
                    "id": "run",
                    "type": "llm",
                    "title": "Run",
                    "inputs": {"prompt": "Analyze ${setup.outputs.domain}", "agent": "cliver"},
                    "outputs": {},
                    "status": "idle",
                },
            ],
        }
        (d / "template.json").write_text(json.dumps(template))

        yield Path(tmpdir) / "scenarios"


def test_discover_scenarios(scenario_dir):
    from cliver.project.scenario_registry import ScenarioRegistry

    registry = ScenarioRegistry([scenario_dir])
    scenarios = registry.list_scenarios()
    assert len(scenarios) == 1
    assert scenarios[0].id == "test-scenario"
    assert scenarios[0].display_name == "Test Scenario"
    assert scenarios[0].tags == ["test", "demo"]


def test_get_scenario(scenario_dir):
    from cliver.project.scenario_registry import ScenarioRegistry

    registry = ScenarioRegistry([scenario_dir])
    s = registry.get_scenario("test-scenario")
    assert s is not None
    assert s.name == "test-scenario"


def test_get_scenario_not_found(scenario_dir):
    from cliver.project.scenario_registry import ScenarioRegistry

    registry = ScenarioRegistry([scenario_dir])
    assert registry.get_scenario("nonexistent") is None


def test_get_template(scenario_dir):
    from cliver.project.scenario_registry import ScenarioRegistry

    registry = ScenarioRegistry([scenario_dir])
    t = registry.get_template("test-scenario")
    assert t is not None
    assert t["$schema"] == "cliver-lab-v1"
    assert len(t["cells"]) == 2


def test_resolve_issue_refs():
    from cliver.project.scenario_registry import _resolve_issue_refs

    template = {
        "title": "${issue.title}",
        "description": "${issue.description}",
        "cells": [
            {"inputs": {"default": "${issue.title}", "prompt": "Analyze ${setup.outputs.domain}"}},
        ],
    }
    result = _resolve_issue_refs(template, {"title": "AI Research", "description": "Survey papers"})
    assert result["title"] == "AI Research"
    assert result["description"] == "Survey papers"
    assert result["cells"][0]["inputs"]["default"] == "AI Research"
    assert result["cells"][0]["inputs"]["prompt"] == "Analyze ${setup.outputs.domain}"


def test_generate_lab(scenario_dir):
    from cliver.lab.store import LabStore
    from cliver.project.scenario_registry import ScenarioRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LabStore(Path(tmpdir))
        registry = ScenarioRegistry([scenario_dir])
        issue = Issue(
            id="iss_test",
            project_id="proj_test",
            title="Transformer Survey",
            description="Survey recent papers",
        )
        lab = registry.generate_lab("test-scenario", issue, store)
        assert lab is not None
        assert lab.title == "Transformer Survey"
        assert lab.description == "Survey recent papers"
        assert lab.scenario_id == "test-scenario"
        assert len(lab.cells) == 2
        assert lab.cells[0].inputs["schema"]["domain"]["default"] == "Transformer Survey"
        assert "${setup.outputs.domain}" in lab.cells[1].inputs["prompt"]


def test_generate_lab_not_found(scenario_dir):
    from cliver.lab.store import LabStore
    from cliver.project.scenario_registry import ScenarioRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LabStore(Path(tmpdir))
        registry = ScenarioRegistry([scenario_dir])
        issue = Issue(id="iss_test", project_id="proj_test", title="Test")
        assert registry.generate_lab("nonexistent", issue, store) is None


def test_empty_dirs():
    from cliver.project.scenario_registry import ScenarioRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ScenarioRegistry([Path(tmpdir)])
        assert registry.list_scenarios() == []


def test_builtin_scenario_exists():
    """Test that the builtin research-ai-lab scenario is discoverable."""
    from cliver.project.scenario_registry import ScenarioRegistry

    builtin_dir = Path(__file__).parent.parent / "src" / "cliver" / "scenarios"
    if builtin_dir.exists():
        registry = ScenarioRegistry([builtin_dir])
        scenarios = registry.list_scenarios()
        ids = [s.id for s in scenarios]
        assert "research-ai-lab" in ids


def test_builtin_research_ai_lab_content():
    """Test that research-ai-lab scenario has complete content."""
    from cliver.project.scenario_registry import ScenarioRegistry

    builtin_dir = Path(__file__).parent.parent / "src" / "cliver" / "scenarios"
    if not builtin_dir.exists():
        pytest.skip("builtin scenarios dir not found")

    registry = ScenarioRegistry([builtin_dir])
    s = registry.get_scenario("research-ai-lab")
    assert s is not None
    assert s.display_name == "Research AI Lab"
    assert "research" in s.tags
    assert "cliver" in s.agent_requirements

    template = registry.get_template("research-ai-lab")
    assert template is not None
    assert template["$schema"] == "cliver-lab-v1"

    cells = template["cells"]
    assert len(cells) == 7

    cell_ids = [c["id"] for c in cells]
    assert cell_ids == ["setup", "intro", "search", "search_summary", "analyze", "review", "export_info"]

    cell_types = [c["type"] for c in cells]
    assert cell_types == ["config", "display", "llm", "display", "llm", "llm", "display"]

    setup = cells[0]
    schema = setup["inputs"]["schema"]
    assert "domain" in schema
    assert "year_range" in schema
    assert "max_papers" in schema
    assert "agent" in schema
    assert "review_style" in schema

    search = cells[2]
    assert "${setup.outputs.domain}" in search["inputs"]["prompt"]
    assert search["inputs"]["output_format"] == "json"

    review = cells[5]
    assert "${analyze.outputs.text}" in review["inputs"]["prompt"]
    assert "${setup.outputs.review_style}" in review["inputs"]["prompt"]


def test_builtin_research_ai_lab_generates_lab():
    """Test that research-ai-lab scenario generates a valid lab."""
    from cliver.lab.store import LabStore
    from cliver.project.scenario_registry import ScenarioRegistry

    builtin_dir = Path(__file__).parent.parent / "src" / "cliver" / "scenarios"
    if not builtin_dir.exists():
        pytest.skip("builtin scenarios dir not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LabStore(Path(tmpdir))
        registry = ScenarioRegistry([builtin_dir])
        issue = Issue(
            id="iss_test",
            project_id="proj_test",
            title="Transformer Architectures",
            description="Survey recent advances in transformer models",
        )
        lab = registry.generate_lab("research-ai-lab", issue, store)
        assert lab is not None
        assert lab.title == "Transformer Architectures"
        assert lab.description == "Survey recent advances in transformer models"
        assert lab.scenario_id == "research-ai-lab"
        assert len(lab.cells) == 7

        setup = lab.get_cell("setup")
        assert setup.inputs["schema"]["domain"]["default"] == "AI Agent Architectures"

        search = lab.get_cell("search")
        assert "${setup.outputs.domain}" in search.inputs["prompt"]
