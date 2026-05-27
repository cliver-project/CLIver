"""Tests for ScenarioInstaller."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from cliver.project.scenario_registry import ScenarioRegistry
from cliver.scenario_installer import ScenarioInstaller


@pytest.fixture
def setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        user_dir = Path(tmpdir) / "scenarios"
        user_dir.mkdir()
        registry = ScenarioRegistry([user_dir])
        installer = ScenarioInstaller(user_dir, registry)
        yield installer, registry, user_dir


def _create_scenario_dir(path: Path, name: str = "test-scenario"):
    d = path / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "display_name": f"Test {name}", "tags": ["test"]}
    (d / "scenario.yaml").write_text(yaml.dump(meta))
    template = {"$schema": "cliver-lab-v1", "title": "Test", "cells": []}
    (d / "template.json").write_text(json.dumps(template))
    return d


def test_validate_valid(setup):
    installer, _, user_dir = setup
    d = _create_scenario_dir(user_dir.parent, "valid")
    valid, error = installer.validate_scenario_dir(d)
    assert valid is True
    assert error == ""


def test_validate_missing_yaml(setup):
    installer, _, user_dir = setup
    d = user_dir.parent / "bad"
    d.mkdir()
    (d / "template.json").write_text('{"$schema": "cliver-lab-v1"}')
    valid, error = installer.validate_scenario_dir(d)
    assert valid is False
    assert "scenario.yaml" in error


def test_validate_missing_template(setup):
    installer, _, user_dir = setup
    d = user_dir.parent / "bad2"
    d.mkdir()
    (d / "scenario.yaml").write_text(yaml.dump({"name": "x", "display_name": "X"}))
    valid, error = installer.validate_scenario_dir(d)
    assert valid is False
    assert "template.json" in error


def test_validate_missing_required_field(setup):
    installer, _, user_dir = setup
    d = user_dir.parent / "bad3"
    d.mkdir()
    (d / "scenario.yaml").write_text(yaml.dump({"name": "x"}))
    (d / "template.json").write_text('{"$schema": "cliver-lab-v1"}')
    valid, error = installer.validate_scenario_dir(d)
    assert valid is False
    assert "display_name" in error


def test_validate_wrong_schema(setup):
    installer, _, user_dir = setup
    d = user_dir.parent / "bad4"
    d.mkdir()
    (d / "scenario.yaml").write_text(yaml.dump({"name": "x", "display_name": "X"}))
    (d / "template.json").write_text('{"$schema": "wrong"}')
    valid, error = installer.validate_scenario_dir(d)
    assert valid is False
    assert "schema" in error.lower()


def test_validate_unsafe_file(setup):
    installer, _, user_dir = setup
    d = _create_scenario_dir(user_dir.parent, "unsafe")
    # Write a python file to test unsafe file detection (intentionally malicious for testing)
    (d / "malicious.py").write_text("print('test')")
    valid, error = installer.validate_scenario_dir(d)
    assert valid is False
    assert "Unsafe" in error


def test_remove_installed(setup):
    installer, registry, user_dir = setup
    _create_scenario_dir(user_dir, "removable")
    registry.refresh()
    assert installer.remove("removable") is True
    assert not (user_dir / "removable").exists()


def test_remove_not_found(setup):
    installer, _, _ = setup
    assert installer.remove("nonexistent") is False


def test_remove_builtin_rejected(setup):
    installer, registry, user_dir = setup
    from cliver.project.models import Scenario

    registry._scenarios["builtin-one"] = Scenario(
        id="builtin-one",
        name="builtin-one",
        display_name="Builtin",
        source="builtin",
    )
    with pytest.raises(ValueError, match="builtin"):
        installer.remove("builtin-one")


def test_invalid_source_format(setup):
    installer, _, _ = setup
    with pytest.raises(ValueError, match="Invalid source"):
        installer.install_from_github("not-github:something")


def test_invalid_github_path(setup):
    installer, _, _ = setup
    with pytest.raises(ValueError, match="Expected"):
        installer.install_from_github("github:only-user")


def test_already_installed(setup):
    installer, _, user_dir = setup
    _create_scenario_dir(user_dir, "existing")
    with pytest.raises(ValueError, match="already installed"):
        installer.install_from_github("github:user/existing")
