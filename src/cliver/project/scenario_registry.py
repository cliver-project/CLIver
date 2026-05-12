"""ScenarioRegistry — discovers and manages scenario templates."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from cliver.project.models import Issue, Scenario

if TYPE_CHECKING:
    from cliver.notebook.models import Notebook
    from cliver.notebook.store import NotebookStore

logger = logging.getLogger(__name__)

_ISSUE_REF_PATTERN = re.compile(r"\$\{issue\.(\w+)\}")


class ScenarioRegistry:
    """Discovers scenario templates from filesystem directories."""

    def __init__(self, dirs: List[Path]):
        self._dirs = dirs
        self._scenarios: Dict[str, Scenario] = {}
        self._discover()

    def _discover(self) -> None:
        self._scenarios.clear()
        for d in self._dirs:
            if not d.exists():
                continue
            for scenario_dir in sorted(d.iterdir()):
                if not scenario_dir.is_dir():
                    continue
                meta_path = scenario_dir / "scenario.yaml"
                if not meta_path.exists():
                    continue
                try:
                    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                    scenario = Scenario(
                        id=scenario_dir.name,
                        name=meta.get("name", scenario_dir.name),
                        display_name=meta.get("display_name", scenario_dir.name),
                        description=meta.get("description", ""),
                        tags=meta.get("tags", []),
                        agent_requirements=meta.get("agent_requirements", []),
                        source="builtin" if "scenarios" in str(d) else "user",
                        path=str(scenario_dir),
                    )
                    self._scenarios[scenario.id] = scenario
                except Exception as e:
                    logger.warning("Failed to load scenario %s: %s", scenario_dir.name, e)

    def list_scenarios(self) -> List[Scenario]:
        return list(self._scenarios.values())

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        return self._scenarios.get(scenario_id)

    def get_template(self, scenario_id: str) -> Optional[dict]:
        scenario = self._scenarios.get(scenario_id)
        if not scenario or not scenario.path:
            return None
        template_path = Path(scenario.path) / "template.json"
        if not template_path.exists():
            return None
        try:
            return json.loads(template_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load template for %s: %s", scenario_id, e)
            return None

    def generate_notebook(
        self,
        scenario_id: str,
        issue: Issue,
        notebook_store: "NotebookStore",
    ) -> Optional["Notebook"]:
        template = self.get_template(scenario_id)
        if not template:
            return None

        issue_vars = {
            "title": issue.title,
            "description": issue.description,
        }

        resolved = _resolve_issue_refs(template, issue_vars)

        title = resolved.get("title", issue.title)
        description = resolved.get("description", issue.description)
        default_agent = resolved.get("default_agent")
        cells = resolved.get("cells", [])

        notebook = notebook_store.create(
            title=title,
            description=description,
            scenario_id=scenario_id,
            default_agent=default_agent,
            cells=cells,
        )
        return notebook

    def refresh(self) -> None:
        self._discover()


def _resolve_issue_refs(obj: Any, issue_vars: Dict[str, str]) -> Any:
    """Resolve ${issue.*} placeholders in a template.

    Only resolves issue-time refs (${issue.title}, ${issue.description}).
    Preserves runtime refs (${cell_id.outputs.*}).
    """
    if isinstance(obj, str):
        def replacer(match: re.Match) -> str:
            field = match.group(1)
            return issue_vars.get(field, match.group(0))
        return _ISSUE_REF_PATTERN.sub(replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_issue_refs(v, issue_vars) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_issue_refs(item, issue_vars) for item in obj]
    return obj
