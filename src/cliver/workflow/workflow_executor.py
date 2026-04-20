"""
Workflow Executor — thin wrapper over LangGraph StateGraph execution.

Loads workflow YAML, compiles to a LangGraph graph via WorkflowCompiler,
executes with SQLite checkpointing for pause/resume support.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from cliver.llm import AgentCore
from cliver.workflow.compiler import WorkflowCompiler
from cliver.workflow.persistence import WorkflowStore
from cliver.workflow.subagent_factory import SubAgentFactory

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Executes workflows using LangGraph with subagent isolation."""

    def __init__(
        self,
        task_executor: AgentCore,
        store: WorkflowStore,
        db_path: Optional[Path] = None,
        app_config=None,
        skill_manager=None,
    ):
        self.task_executor = task_executor
        self.store = store
        self.compiler = WorkflowCompiler()

        self._db_path = db_path
        self._checkpointer = None

        self._subagent_factory = None
        if app_config and skill_manager:
            self._subagent_factory = SubAgentFactory(app_config, skill_manager)

        self._app_config = app_config
        self._skill_manager = skill_manager

    async def _get_checkpointer(self):
        """Lazy-init the async SQLite checkpointer."""
        if self._checkpointer is None and self._db_path:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self._checkpointer = AsyncSqliteSaver.from_conn_string(str(self._db_path))
            await self._checkpointer.setup()
        return self._checkpointer

    async def execute_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a workflow by name.

        Returns the final LangGraph state dict, or None if workflow not found.
        """
        workflow = self.store.load_workflow(workflow_name)
        if not workflow:
            logger.error("Workflow '%s' not found", workflow_name)
            return None

        checkpointer = await self._get_checkpointer()
        graph = self.compiler.compile(workflow, checkpointer=checkpointer)

        execution_id = execution_id or str(uuid.uuid4())[:8]
        outputs_dir = str(Path(workflow.outputs_dir or f".cliver/workflow-runs/{workflow_name}") / execution_id)

        config = {
            "configurable": {
                "thread_id": f"{workflow_name}_{execution_id}",
                "task_executor": self.task_executor,
                "subagent_factory": self._subagent_factory,
                "workflow_store": self.store,
                "db_path": self._db_path,
                "app_config": self._app_config,
                "skill_manager": self._skill_manager,
            }
        }

        initial_state = {
            "inputs": workflow.get_initial_inputs(inputs),
            "steps": {},
            "outputs_dir": outputs_dir,
            "workflow_name": workflow_name,
            "execution_id": execution_id,
            "error": None,
        }

        logger.info("Executing workflow '%s' (id: %s)", workflow_name, execution_id)
        result = await graph.ainvoke(initial_state, config)
        return result

    async def resume_workflow(
        self,
        workflow_name: str,
        thread_id: str,
        resume_value: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Resume a paused workflow from its checkpoint."""
        from langgraph.types import Command

        workflow = self.store.load_workflow(workflow_name)
        if not workflow:
            return None

        checkpointer = await self._get_checkpointer()
        graph = self.compiler.compile(workflow, checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "task_executor": self.task_executor,
                "subagent_factory": self._subagent_factory,
                "workflow_store": self.store,
                "db_path": self._db_path,
                "app_config": self._app_config,
                "skill_manager": self._skill_manager,
            }
        }

        if resume_value is not None:
            result = await graph.ainvoke(Command(resume=resume_value), config)
        else:
            result = await graph.ainvoke(None, config)

        return result
