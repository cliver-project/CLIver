"""
YAML-based persistence for workflow definitions.

Workflows stored as {workflows_dir}/{name}.yaml
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from cliver.workflow.workflow_models import Workflow

logger = logging.getLogger(__name__)


class WorkflowStore:
    """Manages workflow YAML files and execution state."""

    def __init__(self, workflows_dir: Path):
        self.workflows_dir = Path(workflows_dir)

    def _ensure_dir(self) -> None:
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    # -- Workflow CRUD ---------------------------------------------------------

    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow definition as YAML."""
        self._ensure_dir()
        path = self.workflows_dir / f"{workflow.name}.yaml"
        data = workflow.model_dump(exclude_none=True, mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def load_workflow(self, name: str) -> Optional[Workflow]:
        """Load a workflow definition from YAML."""
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return Workflow(**data)
        except Exception as e:
            logger.error(f"Failed to load workflow '{name}': {e}")
            return None

    def list_workflows(self) -> List[str]:
        """List all saved workflow names."""
        if not self.workflows_dir.is_dir():
            return []
        return sorted(p.stem for p in self.workflows_dir.glob("*.yaml") if not p.stem.endswith(".state"))

    def delete_workflow(self, name: str) -> bool:
        """Delete a workflow."""
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            return False
        path.unlink()
        return True
