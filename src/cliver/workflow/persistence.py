"""
YAML-based persistence for workflow definitions and execution state.

Workflows stored as {workflows_dir}/{name}.yaml
Execution state as {workflows_dir}/{name}.state.yaml
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from cliver.workflow.workflow_models import Workflow, WorkflowExecutionState

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
        """Delete a workflow and its state."""
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            return False
        path.unlink()
        state_path = self.workflows_dir / f"{name}.state.yaml"
        if state_path.exists():
            state_path.unlink()
        return True

    # -- Execution state -------------------------------------------------------

    def save_state(self, state: WorkflowExecutionState) -> None:
        """Save execution state as YAML."""
        self._ensure_dir()
        path = self.workflows_dir / f"{state.workflow_name}.state.yaml"
        data = state.model_dump(exclude_none=True, mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def load_state(self, workflow_name: str) -> Optional[WorkflowExecutionState]:
        """Load the last execution state for a workflow."""
        path = self.workflows_dir / f"{workflow_name}.state.yaml"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return WorkflowExecutionState(**data)
        except Exception as e:
            logger.error(f"Failed to load state for '{workflow_name}': {e}")
            return None
