"""
Local file-based cache persistence for workflow engine.

No threading locks — CLIver workflows run in a single async context.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from cliver.workflow.persistence.base import CacheProvider
from cliver.workflow.workflow_models import WorkflowExecutionState

logger = logging.getLogger(__name__)


class LocalCacheProvider(CacheProvider):
    """Local file-based cache for workflow execution state.

    Directory structure:
        {cache_dir}/{workflow_name}/{execution_id}/state.json
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
            cache_dir = os.path.join(cache_home, "cliver")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_execution_dir(self, workflow_name: str, execution_id: str) -> Path:
        return self.cache_dir / workflow_name / execution_id

    def save_execution_state(self, state: WorkflowExecutionState) -> bool:
        """Save workflow execution state to JSON file."""
        try:
            execution_dir = self._get_execution_dir(state.workflow_name, state.execution_id)
            execution_dir.mkdir(parents=True, exist_ok=True)
            cache_file = execution_dir / "state.json"
            with open(cache_file, "w") as f:
                json.dump(state.model_dump(), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save execution state: {e}")
            raise

    def load_execution_state(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """Load workflow execution state from JSON file."""
        cache_file = self._get_execution_dir(workflow_name, execution_id) / "state.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r") as f:
                return WorkflowExecutionState(**json.load(f))
        except Exception as e:
            logger.error(f"Failed to load execution state: {e}")
            raise

    def remove_execution_state(self, workflow_name: str, execution_id: str) -> bool:
        """Remove a workflow execution directory."""
        execution_dir = self._get_execution_dir(workflow_name, execution_id)
        if not execution_dir.exists():
            return False
        shutil.rmtree(execution_dir)
        return True

    def list_executions(self, workflow_name: str) -> Dict[str, Dict[str, Any]]:
        """List all cached executions for a workflow."""
        executions = {}
        workflow_dir = self.cache_dir / workflow_name
        if not workflow_dir.is_dir():
            return executions

        for execution_dir in workflow_dir.iterdir():
            if not execution_dir.is_dir():
                continue
            state_file = execution_dir / "state.json"
            if state_file.exists():
                try:
                    with open(state_file, "r") as f:
                        data = json.load(f)
                    executions[execution_dir.name] = {
                        "workflow_name": data.get("workflow_name"),
                        "status": data.get("status"),
                        "current_step_index": data.get("current_step_index"),
                        "completed_steps": data.get("completed_steps", []),
                    }
                except Exception as e:
                    logger.warning(f"Failed to read {state_file}: {e}")
        return executions

    def clear_all_executions(self, workflow_name: str) -> int:
        """Clear all cached executions for a workflow."""
        workflow_dir = self.cache_dir / workflow_name
        if workflow_dir.is_dir():
            shutil.rmtree(workflow_dir)
            return 1
        return 0

    def get_execution_cache_dir(self, workflow_name: str, execution_id: str) -> Optional[str]:
        """Get the cache directory path for a workflow execution."""
        if not workflow_name or not execution_id:
            return None
        return str(self._get_execution_dir(workflow_name, execution_id))
