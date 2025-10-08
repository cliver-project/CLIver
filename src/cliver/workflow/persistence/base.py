"""
Abstract base classes for Cliver workflow persistence providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from cliver.workflow.workflow_models import WorkflowExecutionState, ExecutionResult


class PersistenceProvider(ABC):
    """Abstract base class for workflow execution state persistence providers."""

    @abstractmethod
    def save_execution_state(self, state: WorkflowExecutionState) -> bool:
        """Save workflow execution state.

        Args:
            state: Workflow execution state to save

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_execution_state(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """Load workflow execution state.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID to load

        Returns:
            WorkflowExecutionState if found, None otherwise
        """
        pass

    @abstractmethod
    def remove_execution_state(self, workflow_name: str, execution_id: str) -> bool:
        """Remove workflow execution state.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID to remove

        Returns:
            True if removed, False if not found or error occurred
        """
        pass

    @abstractmethod
    def save_step_result(self, workflow_name: str, execution_id: str, step_id: str, result: ExecutionResult) -> bool:
        """Save step execution result to cache.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID
            step_id: Step ID
            result: Step execution result to save

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_step_result(self, workflow_name: str, execution_id: str, step_id: str) -> Optional[ExecutionResult]:
        """Load step execution result from cache.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID
            step_id: Step ID

        Returns:
            Step execution result if found, None otherwise
        """
        pass


class CacheProvider(PersistenceProvider):
    """Abstract base class for cache-based persistence providers."""

    @abstractmethod
    def list_executions(self, workflow_name: str) -> Dict[str, Dict[str, Any]]:
        """List all cached workflow executions.

        Arguments:
            workflow_name: Name of the workflow

        Returns:
            Dictionary mapping execution IDs to execution metadata
        """
        pass

    @abstractmethod
    def clear_all_executions(self, workflow_name: str) -> int:
        """Clear all cached workflow executions.

        Arguments:
            workflow_name: Name of the workflow

        Returns:
            Number of executions cleared
        """
        pass