"""
Local cache persistence for Cliver workflow engine.
"""
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any
from cliver.workflow.workflow_models import WorkflowExecutionState, ExecutionResult
from cliver.workflow.persistence.base import CacheProvider

logger = logging.getLogger(__name__)

class LocalCacheProvider(CacheProvider):
    """Thread-safe local file-based cache for workflow execution state.

    Each workflow execution state is stored in a directory structure:
    {cache_dir}/{workflow_name}/{execution_id}/state.json
    Step results are stored as:
    {cache_dir}/{workflow_name}/{execution_id}/{step_id}_result.json

    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the local cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cache/cliver
        """
        if cache_dir is None:
            # Default to user cache directory
            cache_home = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
            cache_dir = os.path.join(cache_home, 'cliver')

        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dir()
        # Use fine-grained locking per execution_id for better concurrency
        self._locks = OrderedDict()  # Cache for locks
        self._locks_lock = threading.RLock()  # Lock to protect the locks dictionary itself

    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_lock_for_execution(self, workflow_name: str, execution_id: str) -> threading.RLock:
        """Get or create a lock for a specific workflow_name and execution_id combination."""
        # Create composite key
        composite_key = f"{workflow_name}:{execution_id}"
        with self._locks_lock:
            if composite_key in self._locks:
                # Move to end (most recently used)
                lock = self._locks.pop(composite_key)
                self._locks[composite_key] = lock
                return lock
            else:
                # Create new lock
                lock = threading.RLock()
                self._locks[composite_key] = lock
                return lock

    def _get_execution_dir(self, workflow_name: str, execution_id: str) -> Path:
        """Get the execution directory path.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID

        Returns:
            Path to the execution directory
        """
        return self.cache_dir / workflow_name / execution_id

    def save_execution_state(self, state: WorkflowExecutionState) -> bool:
        """Thread-safe save workflow execution state to cache.

        Args:
            state: Workflow execution state to save

        Returns:
            True if saved successfully, False otherwise
        """
        if not state.execution_id or len(state.execution_id.strip()) == 0:
            raise Exception(f"Workflow execution state: {str(state.model_dump())} has an empty execution_id")
        with self._get_lock_for_execution(state.workflow_name, state.execution_id):
            try:
                # Create execution directory
                execution_dir = self._get_execution_dir(state.workflow_name, state.execution_id)
                execution_dir.mkdir(parents=True, exist_ok=True)

                # Create cache file path
                cache_file = execution_dir / "state.json"

                # Convert state to dict for JSON serialization
                state_dict = state.model_dump()

                # Write to file
                with open(cache_file, 'w') as f:
                    json.dump(state_dict, f, indent=2, default=str)

                logger.debug(f"Saved execution state to {cache_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to save execution state: {e}")
                raise e

    def load_execution_state(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """Thread-safe load workflow execution state from cache.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID to load

        Returns:
            WorkflowExecutionState if found, None otherwise
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return None
        if not execution_id or len(execution_id.strip()) == 0:
            return None
        with self._get_lock_for_execution(workflow_name, execution_id):
            try:
                # Create cache file path
                cache_file = self._get_execution_dir(workflow_name, execution_id) / "state.json"

                # Check if file exists
                if not cache_file.exists():
                    return None

                # Read from file
                with open(cache_file, 'r') as f:
                    state_dict = json.load(f)

                # Convert dict to WorkflowExecutionState
                state = WorkflowExecutionState(**state_dict)
                logger.debug(f"Loaded execution state from {cache_file}")
                return state
            except Exception as e:
                logger.error(f"Failed to load execution state: {e}")
                raise e

    def remove_execution_state(self, workflow_name: str, execution_id: str) -> bool:
        """Thread-safe remove workflow execution state from cache.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID to remove

        Returns:
            True if removed, False if not found or error occurred
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return False
        if not execution_id or len(execution_id.strip()) == 0:
            return False
        try:
            with self._get_lock_for_execution(workflow_name, execution_id):
                try:
                    # Create execution directory path
                    execution_dir = self._get_execution_dir(workflow_name, execution_id)

                    # Check if directory exists
                    if not execution_dir.exists():
                        return False

                    # Remove entire execution directory
                    import shutil
                    shutil.rmtree(execution_dir)
                    logger.debug(f"Removed execution directory {execution_dir}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to remove execution state: {e}")
                    raise e
        finally:
            # delete the lock if any
            with self._locks_lock:
                composite_key = f"{workflow_name}:{execution_id}"
                self._locks.pop(composite_key, None)

    def list_executions(self, workflow_name: str) -> Dict[str, Dict[str, Any]]:
        """Thread-safe list all cached workflow executions.

        Arguments:
            workflow_name: Name of the workflow

        Returns:
            Dictionary mapping execution IDs to execution metadata
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return {}
        # Use global lock for listing operations since it accesses multiple files
        with self._locks_lock:
            executions = {}
            try:
                # Look for workflow directory
                workflow_dir = self.cache_dir / workflow_name
                if workflow_dir.is_dir():
                    # Look for all execution directories within the workflow
                    for execution_dir in workflow_dir.iterdir():
                        if execution_dir.is_dir():
                            execution_id = execution_dir.name
                            try:
                                # Look for state.json file
                                state_file = execution_dir / "state.json"
                                if state_file.exists():
                                    with open(state_file, 'r') as f:
                                        state_dict = json.load(f)

                                    executions[execution_id] = {
                                        'workflow_name': state_dict.get('workflow_name'),
                                        'status': state_dict.get('status'),
                                        'current_step_index': state_dict.get('current_step_index'),
                                        'completed_steps': state_dict.get('completed_steps', []),
                                    }
                            except Exception as e:
                                logger.warning(f"Failed to read execution state from {execution_dir}: {e}")
                                continue
            except Exception as e:
                logger.error(f"Failed to list executions: {e}")
                raise e

            return executions

    def clear_all_executions(self, workflow_name: str) -> int:
        """Thread-safe clear all cached workflow executions.

        Arguments:
            workflow_name: Name of the workflow

        Returns:
            Number of executions cleared
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return 0
        # Use global lock for clearing operations since it affects multiple files
        with self._locks_lock:
            count = 0
            try:
                # Remove workflow directory
                workflow_dir = self.cache_dir / workflow_name
                if workflow_dir.is_dir():
                    try:
                        import shutil
                        shutil.rmtree(workflow_dir)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove workflow directory {workflow_dir}: {e}")
            except Exception as e:
                logger.error(f"Failed to clear executions: {e}")
                raise e
            return count

    def get_execution_cache_dir(self, workflow_name: str, execution_id: str) -> Optional[str]:
        """Get the cache directory path for a workflow execution.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID

        Returns:
            Path to the cache directory for this workflow execution
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return None
        if not execution_id or len(execution_id.strip()) == 0:
            return None
        return str(self._get_execution_dir(workflow_name, execution_id))

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
        if not workflow_name or len(workflow_name.strip()) == 0:
            return False
        if not execution_id or len(execution_id.strip()) == 0:
            return False
        if not step_id or len(step_id.strip()) == 0:
            return False
        if not result:
            return False
        with self._get_lock_for_execution(workflow_name, execution_id):
            try:
                # Create execution directory
                execution_dir = self._get_execution_dir(workflow_name, execution_id)
                execution_dir.mkdir(parents=True, exist_ok=True)

                # Create step result file path
                result_file = execution_dir / f"{step_id}_result.json"

                # Convert result to dict for JSON serialization
                result_dict = result.model_dump()

                # Write to file
                with open(result_file, 'w') as f:
                    json.dump(result_dict, f, indent=2, default=str)

                logger.debug(f"Saved step result to {result_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to save step result: {e}")
                raise e

    def load_step_result(self, workflow_name: str, execution_id: str, step_id: str) -> Optional[ExecutionResult]:
        """Load step execution result from cache.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution ID
            step_id: Step ID

        Returns:
            Step execution result if found, None otherwise
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return None
        if not execution_id or len(execution_id.strip()) == 0:
            return None
        if not step_id or len(step_id.strip()) == 0:
            return None
        # Use execution_id-specific lock for better concurrency
        with self._get_lock_for_execution(workflow_name, execution_id):
            try:
                # Create step result file path
                result_file = self._get_execution_dir(workflow_name, execution_id) / f"{step_id}_result.json"

                # Check if file exists
                if not result_file.exists():
                    return None

                # Read from file
                with open(result_file, 'r') as f:
                    result_dict = json.load(f)

                # Convert dict to ExecutionResult
                result = ExecutionResult(**result_dict)

                logger.debug(f"Loaded step result from {result_file}")
                return result
            except Exception as e:
                logger.error(f"Failed to load step result: {e}")
                raise e
