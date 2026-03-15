"""
Local directory-based workflow manager for CLIver workflow engine.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from cliver.util import get_config_dir
from cliver.workflow.workflow_manager_base import WorkflowManager
from cliver.workflow.workflow_models import Workflow

logger = logging.getLogger(__name__)


def _get_default_workflow_dirs() -> List[Path]:
    """Default workflow directories in priority order."""
    return [
        Path.cwd() / ".cliver" / "workflows",
        get_config_dir() / "workflows",
    ]


def _load_workflow_raw(workflow_file: Path) -> Optional[Workflow]:
    """Load a workflow from a YAML file."""
    if not workflow_file.exists():
        return None
    with open(workflow_file, "r") as f:
        data = yaml.safe_load(f)
    return Workflow(**data)


def _load_workflows_from_directory(directory: Path) -> Dict[str, Workflow]:
    """Load all workflows from a directory."""
    workflows = {}
    for ext in ["*.yaml", "*.yml"]:
        for path in directory.glob(ext):
            try:
                workflow = _load_workflow_raw(path)
                if workflow and workflow.name not in workflows:
                    workflows[workflow.name] = workflow
            except Exception as e:
                logger.warning(f"Failed to load workflow from {path}: {e}")
                raise
    return workflows


class LocalDirectoryWorkflowManager(WorkflowManager):
    """Local directory-based workflow manager with lazy loading."""

    def __init__(self, workflow_dirs: Optional[List[str]] = None):
        if workflow_dirs is not None:
            self.workflow_dirs = [Path(d) for d in workflow_dirs]
        else:
            self.workflow_dirs = _get_default_workflow_dirs()
        self._workflows: Optional[Dict[str, Workflow]] = None

    def load_workflow(self, workflow_name: Union[str, Path]) -> Optional[Workflow]:
        """Load a workflow by name."""
        return self.list_workflows().get(str(workflow_name))

    def list_workflows(self) -> Dict[str, Workflow]:
        """List all available workflows (lazy-loaded, cached)."""
        if self._workflows is not None:
            return self._workflows

        workflows = {}
        for workflow_dir in self.workflow_dirs:
            if not workflow_dir.exists():
                continue
            for name, workflow in _load_workflows_from_directory(workflow_dir).items():
                if name not in workflows:
                    workflows[name] = workflow
        self._workflows = workflows
        return workflows

    def refresh_workflows(self) -> None:
        """Clear cache to force reload on next access."""
        self._workflows = None
