"""
Abstract base class for Workflow Manager in Cliver workflow engine.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

from cliver.workflow.workflow_models import Workflow


class WorkflowManager(ABC):
    """Abstract base class for workflow managers."""

    @abstractmethod
    def load_workflow(self, workflow_name: Union[str, Path]) -> Optional[Workflow]:
        """Load and validate a workflow by name.

        Args:
            workflow_name: Workflow name

        Returns:
            Loaded Workflow object or None if not found
        """
        pass

    @abstractmethod
    def list_workflows(self) -> Dict[str, Workflow]:
        """List available workflows.

        Returns:
            Dictionary mapping workflow names to Workflow objects
        """
        pass
