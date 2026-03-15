"""
Abstract base classes for workflow persistence providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from cliver.workflow.workflow_models import WorkflowExecutionState


class PersistenceProvider(ABC):
    """Abstract base for workflow execution state persistence."""

    @abstractmethod
    def save_execution_state(self, state: WorkflowExecutionState) -> bool:
        """Save workflow execution state."""
        pass

    @abstractmethod
    def load_execution_state(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """Load workflow execution state."""
        pass

    @abstractmethod
    def remove_execution_state(self, workflow_name: str, execution_id: str) -> bool:
        """Remove workflow execution state."""
        pass


class CacheProvider(PersistenceProvider):
    """Extended persistence with listing and cleanup operations."""

    @abstractmethod
    def list_executions(self, workflow_name: str) -> Dict[str, Dict[str, Any]]:
        """List all cached workflow executions."""
        pass

    @abstractmethod
    def clear_all_executions(self, workflow_name: str) -> int:
        """Clear all cached workflow executions."""
        pass
