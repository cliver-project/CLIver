"""Scenario installer — install/remove/validate community scenarios."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from cliver.project.scenario_registry import ScenarioRegistry

logger = logging.getLogger(__name__)

UNSAFE_EXTENSIONS = {".py", ".sh", ".bat", ".cmd", ".ps1", ".exe"}


class ScenarioInstaller:
    """Install, remove, and validate community scenario templates."""

    def __init__(self, user_scenarios_dir: Path, registry: "ScenarioRegistry"):
        self._dir = user_scenarios_dir
        self._registry = registry
        self._dir.mkdir(parents=True, exist_ok=True)

    def install_from_github(self, source: str) -> str:
        """Install a scenario from github:user/repo format.

        Returns the installed scenario name.
        Raises ValueError on invalid source or validation failure.
        Raises RuntimeError on git clone failure.
        """
        if not source.startswith("github:"):
            raise ValueError(f"Invalid source format: '{source}'. Use: github:user/repo")

        repo_path = source[len("github:") :]
        if "/" not in repo_path or repo_path.count("/") != 1:
            raise ValueError(f"Invalid GitHub reference: '{repo_path}'. Expected: user/repo")

        git_url = f"https://github.com/{repo_path}.git"
        repo_name = repo_path.split("/")[1]

        target_dir = self._dir / repo_name
        if target_dir.exists():
            raise ValueError(
                f"Scenario '{repo_name}' is already installed. Remove it first with: /scenario remove {repo_name}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_clone = Path(tmpdir) / repo_name
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", git_url, str(tmp_clone)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone {git_url}: {e.stderr.strip()}") from e
            except FileNotFoundError:
                raise RuntimeError("git is not installed. Install git to use scenario install.") from None

            git_dir = tmp_clone / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)

            valid, error = self.validate_scenario_dir(tmp_clone)
            if not valid:
                raise ValueError(f"Invalid scenario: {error}")

            shutil.copytree(tmp_clone, target_dir)

        self._registry.refresh()

        scenario = self._registry.get_scenario(repo_name)
        display_name = scenario.display_name if scenario else repo_name
        return display_name

    def remove(self, name: str) -> bool:
        """Remove an installed scenario. Returns True if existed."""
        scenario = self._registry.get_scenario(name)
        if scenario and scenario.source == "builtin":
            raise ValueError(f"Cannot remove builtin scenario '{name}'. Only user-installed scenarios can be removed.")

        target_dir = self._dir / name
        if not target_dir.exists():
            return False

        shutil.rmtree(target_dir)
        self._registry.refresh()
        return True

    def validate_scenario_dir(self, path: Path) -> tuple:
        """Validate a scenario directory. Returns (valid: bool, error: str)."""
        meta_path = path / "scenario.yaml"
        if not meta_path.exists():
            return False, "Missing scenario.yaml"

        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Invalid scenario.yaml: {e}"

        if not isinstance(meta, dict):
            return False, "scenario.yaml must be a YAML mapping"
        if not meta.get("name"):
            return False, "scenario.yaml missing required field: name"
        if not meta.get("display_name"):
            return False, "scenario.yaml missing required field: display_name"

        template_path = path / "template.json"
        if not template_path.exists():
            return False, "Missing template.json"

        try:
            template = json.loads(template_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Invalid template.json: {e}"

        if not isinstance(template, dict):
            return False, "template.json must be a JSON object"
        if template.get("$schema") != "cliver-lab-v1":
            return False, "template.json missing or invalid $schema (expected cliver-lab-v1)"

        for f in path.iterdir():
            if f.is_file() and f.suffix.lower() in UNSAFE_EXTENSIONS:
                return False, f"Unsafe file detected: {f.name} (executable files not allowed)"

        return True, ""
