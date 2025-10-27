"""
Local directory-based implementation of Workflow Manager for Cliver workflow engine.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from cliver.util import get_config_dir
from cliver.workflow.workflow_manager_base import WorkflowManager
from cliver.workflow.workflow_models import Workflow

logger = logging.getLogger(__name__)


def _get_default_workflow_dirs() -> List[Path]:
    """Get the default workflow directories.

    Returns:
        List of default workflow directories in order of preference:
        1. .cliver/workflows in current directory
        2. ~/.config/cliver/workflows
    """
    # First priority: .cliver/workflows in current directory
    current_workflow_dir = Path.cwd() / ".cliver" / "workflows"

    # Second priority: ~/.config/cliver/workflows
    config_dir = get_config_dir()
    default_workflow_dir = config_dir / "workflows"
    return [current_workflow_dir, default_workflow_dir]


def _load_workflow_raw(workflow_file: Union[str, Path]) -> Optional[Workflow]:
    """Load and validate a workflow from a YAML file without path resolution.

    Args:
        workflow_file: Path to the workflow YAML file

    Returns:
        Loaded Workflow object or None if not found

    Raises:
        yaml.YAMLError: If workflow file is not valid YAML
        ValueError: If workflow validation fails
    """
    workflow_path = Path(workflow_file)
    if not workflow_path.exists():
        return None

    with open(workflow_path, "r") as f:
        workflow_data = yaml.safe_load(f)

    # Convert to Workflow model
    workflow = Workflow(**workflow_data)
    return workflow


def _load_workflows_from_directory(directory: Path) -> Dict[str, Workflow]:
    """Load workflows from a directory.

    Args:
        directory: Directory to search for workflows
    """
    workflows = {}
    # Check both .yaml and .yml files
    for ext in ["*.yaml", "*.yml"]:
        for workflow_file in directory.glob(ext):
            logger.debug(f"Found workflow file: {workflow_file}")
            try:
                workflow = _load_workflow_raw(workflow_file)
                # we make the first fit precedence, so .yaml over .yml with the same workflow name
                if not workflows.get(workflow.name, None):
                    workflows[workflow.name] = workflow
            except Exception as e:
                logger.warning(f"Failed to load workflow from {workflow_file}: {e}")
                raise e
    return workflows


class LocalDirectoryWorkflowManager(WorkflowManager):
    """Thread-safe local directory-based workflow manager."""

    def __init__(self, workflow_dirs: Optional[List[str]] = None):
        """Initialize the local directory workflow manager.

        Args:
            workflow_dirs: List of directories to search for workflows.
                          Defaults to .cliver/workflows and ~/.config/cliver/workflows
        """
        # Set up workflow directories
        if workflow_dirs is not None:
            self.workflow_dirs = [Path(d) for d in workflow_dirs]
        else:
            # Default directories
            self.workflow_dirs = _get_default_workflow_dirs()
        # cache the workflows during the execution to load only once
        # users need to refresh explicitly to re-load from directories.
        self._workflows = None
        # Add thread-safe lock
        self._lock = threading.RLock()

    def load_workflow(self, workflow_name: Union[str, Path]) -> Optional[Workflow]:
        """Thread-safe load and validate a workflow by name.

        Args:
            workflow_name: Workflow name

        Returns:
            Loaded Workflow object
        """
        with self._lock:
            _workflows = self.list_workflows()
            return _workflows.get(workflow_name, None)

    def list_workflows(self) -> Dict[str, Workflow]:
        """Thread-safe list available workflows.

        Returns:
            Dictionary mapping workflow names to Workflow objects.
        """
        with self._lock:
            if self._workflows is not None and len(self._workflows) > 0:
                return self._workflows

            workflows = {}
            # Check for workflows in configured directories
            for workflow_dir in self.workflow_dirs:
                logger.debug(f"Checking workflow directory: {workflow_dir}")
                if not workflow_dir.exists():
                    logger.debug(f"Workflow directory does not exist: {workflow_dir}")
                    continue
                _workflows_from_dir = _load_workflows_from_directory(workflow_dir)
                if _workflows_from_dir:
                    for (
                        name,
                        _workflow,
                    ) in _workflows_from_dir.items():  # Fixed iteration
                        if name not in workflows:  # Fixed condition check
                            workflows[name] = _workflow
            self._workflows = workflows
            return workflows

    def refresh_workflows(self) -> None:
        """Thread-safe refresh the workflow cache."""
        with self._lock:
            self._workflows = None
            # Force reload on next access
