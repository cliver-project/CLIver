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
            "$schema": "cliver-notebook-v1",
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
    assert t["$schema"] == "cliver-notebook-v1"
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


def test_generate_notebook(scenario_dir):
    from cliver.project.scenario_registry import ScenarioRegistry
    from cliver.notebook.store import NotebookStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = NotebookStore(Path(tmpdir))
        registry = ScenarioRegistry([scenario_dir])
        issue = Issue(
            id="iss_test", project_id="proj_test",
            title="Transformer Survey", description="Survey recent papers",
        )
        nb = registry.generate_notebook("test-scenario", issue, store)
        assert nb is not None
        assert nb.title == "Transformer Survey"
        assert nb.description == "Survey recent papers"
        assert nb.scenario_id == "test-scenario"
        assert len(nb.cells) == 2
        assert nb.cells[0].inputs["schema"]["domain"]["default"] == "Transformer Survey"
        assert "${setup.outputs.domain}" in nb.cells[1].inputs["prompt"]


def test_generate_notebook_not_found(scenario_dir):
    from cliver.project.scenario_registry import ScenarioRegistry
    from cliver.notebook.store import NotebookStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = NotebookStore(Path(tmpdir))
        registry = ScenarioRegistry([scenario_dir])
        issue = Issue(id="iss_test", project_id="proj_test", title="Test")
        assert registry.generate_notebook("nonexistent", issue, store) is None


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
