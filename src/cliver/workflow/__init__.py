"""
Cliver Workflow Engine Package.

This package provides a modular, extensible workflow engine for AI and automation tasks.
"""

# Import key components for easy access
from cliver.workflow.persistence import LocalCacheProvider
from cliver.workflow.workflow_manager_base import WorkflowManager
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager
from cliver.workflow.workflow_models import (
    ExecutionContext,
    ExecutionResult,
    StepType,
    Workflow,
)

__all__ = [
    "WorkflowManager",
    "LocalDirectoryWorkflowManager",
    "Workflow",
    "StepType",
    "ExecutionContext",
    "ExecutionResult",
    "LocalCacheProvider",
]
