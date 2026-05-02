"""
Workflow Executor — thin wrapper over LangGraph StateGraph execution.

Loads workflow YAML, compiles to a LangGraph graph via WorkflowCompiler,
executes with SQLite checkpointing for pause/resume support.
"""

import logging
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.llm import AgentCore
from cliver.workflow.compiler import WorkflowCompiler
from cliver.workflow.persistence import WorkflowStore
from cliver.workflow.subagent_factory import SubAgentFactory
from cliver.workflow.workflow_models import Workflow

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Executes workflows using LangGraph with subagent isolation."""

    def __init__(
        self,
        agent_core: AgentCore,
        store: WorkflowStore,
        db_path: Optional[Path] = None,
        app_config=None,
        skill_manager=None,
    ):
        self.agent_core = agent_core
        self.store = store
        self.compiler = WorkflowCompiler()

        self._db_path = db_path
        self._checkpointer = None
        self._exit_stack = AsyncExitStack()

        self._subagent_factory = None
        if app_config and skill_manager:
            self._subagent_factory = SubAgentFactory(app_config, skill_manager, agent_name=agent_core.agent_name)

        self._app_config = app_config
        self._skill_manager = skill_manager

    async def close(self):
        """Clean up all managed resources."""
        await self._exit_stack.aclose()
        self._checkpointer = None

    async def _get_checkpointer(self):
        """Lazy-init the async SQLite checkpointer."""
        if self._checkpointer is None and self._db_path:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self._checkpointer = await self._exit_stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(self._db_path))
            )
        return self._checkpointer

    async def execute_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        workflow_file: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a workflow by name, falling back to file path.

        Resolution order:
        1. Try loading by *workflow_name* from the store.
        2. If not found and *workflow_file* is provided, load from file.
        3. If still not found, return None.
        """
        workflow = self.store.load_workflow(workflow_name)
        base_dir = str(self.store.workflows_dir)

        if not workflow and workflow_file:
            workflow = WorkflowStore.load_workflow_from_file(workflow_file)
            if workflow:
                file_path = Path(workflow_file)
                base_dir = str(file_path.parent.resolve()) if file_path.is_absolute() else base_dir

        if not workflow:
            logger.error("Workflow '%s' not found", workflow_name)
            return None

        return await self.execute_workflow_obj(workflow, inputs=inputs, execution_id=execution_id, base_dir=base_dir)

    async def execute_workflow_obj(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a pre-loaded Workflow object directly."""
        checkpointer = await self._get_checkpointer()
        effective_base = base_dir or str(self.store.workflows_dir)
        graph = self.compiler.compile(workflow, checkpointer=checkpointer, store=self.store, base_dir=effective_base)

        execution_id = execution_id or str(uuid.uuid4())[:8]
        default_runs_dir = self.store.workflows_dir.parent / "workflow-runs" / workflow.name
        outputs_dir = str(Path(workflow.outputs_dir or str(default_runs_dir)) / execution_id)

        config = {
            "configurable": {
                "thread_id": f"{workflow.name}_{execution_id}",
                "agent_core": self.agent_core,
                "subagent_factory": self._subagent_factory,
                "workflow_store": self.store,
                "db_path": self._db_path,
                "app_config": self._app_config,
                "skill_manager": self._skill_manager,
                "workflow_base_dir": base_dir or str(self.store.workflows_dir),
            }
        }

        initial_state = {
            "inputs": workflow.get_initial_inputs(inputs),
            "steps": {},
            "outputs_dir": outputs_dir,
            "workflow_name": workflow.name,
            "execution_id": execution_id,
            "error": None,
        }

        thread_id = f"{workflow.name}_{execution_id}"
        if self._db_path:
            WorkflowStore.record_execution_start(
                Path(self._db_path),
                thread_id,
                workflow.name,
                execution_id,
                inputs=initial_state["inputs"],
            )

        logger.info("Executing workflow '%s' (id: %s)", workflow.name, execution_id)
        try:
            result = await graph.ainvoke(initial_state, config)
            if self._db_path:
                status = "failed" if result and result.get("error") else "completed"
                WorkflowStore.record_execution_end(
                    Path(self._db_path),
                    thread_id,
                    status=status,
                    error=result.get("error") if result else None,
                )
            return result
        except Exception as e:
            if self._db_path:
                WorkflowStore.record_execution_end(
                    Path(self._db_path),
                    thread_id,
                    status="failed",
                    error=str(e),
                )
            raise

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
        graph = self.compiler.compile(
            workflow, checkpointer=checkpointer, store=self.store, base_dir=str(self.store.workflows_dir)
        )

        config = self._make_config(thread_id)
        error = await self._validate_checkpoint_compatibility(graph, workflow, config)
        if error:
            return {"error": error, "steps": {}}

        if resume_value is not None:
            result = await graph.ainvoke(Command(resume=resume_value), config)
        else:
            result = await graph.ainvoke(None, config)

        return result

    def _make_config(self, thread_id: str) -> dict:
        """Build a LangGraph config dict for a given thread_id."""
        return {
            "configurable": {
                "thread_id": thread_id,
                "agent_core": self.agent_core,
                "subagent_factory": self._subagent_factory,
                "workflow_store": self.store,
                "db_path": self._db_path,
                "app_config": self._app_config,
                "skill_manager": self._skill_manager,
                "workflow_base_dir": str(self.store.workflows_dir),
            }
        }

    async def _validate_checkpoint_compatibility(self, graph, workflow: Workflow, config: dict) -> Optional[str]:
        """Check if a checkpoint is compatible with the current workflow structure.

        Compares step IDs and dependency edges between the checkpoint state
        and the current workflow definition. Returns an error message if
        incompatible, None if compatible.

        Prompt/model changes are fine — only structural changes are rejected.
        """
        try:
            snapshot = await graph.aget_state(config)
        except Exception:
            return None  # no checkpoint yet, nothing to validate

        if not snapshot or not snapshot.values:
            return None

        checkpoint_steps = set(snapshot.values.get("steps", {}).keys())
        if not checkpoint_steps:
            return None

        workflow_steps = {s.id for s in workflow.steps}

        removed = checkpoint_steps - workflow_steps
        if removed:
            return (
                f"Workflow structure changed: steps {removed} exist in the checkpoint "
                f"but were removed from the workflow. Cannot resume — start a new execution "
                f"or delete this execution with: /workflow delete-execution {config['configurable']['thread_id']}"
            )

        # Check that completed steps still have the same dependencies
        for step in workflow.steps:
            if step.id in checkpoint_steps:
                for dep in step.depends_on:
                    if dep not in checkpoint_steps and dep not in workflow_steps:
                        return (
                            f"Workflow structure changed: step '{step.id}' now depends on "
                            f"'{dep}' which doesn't exist. Cannot resume."
                        )

        added = workflow_steps - checkpoint_steps
        if added:
            # New steps are fine if they depend on steps that haven't run yet
            for step in workflow.steps:
                if step.id in added and not step.depends_on:
                    # New root step — would run from scratch alongside checkpoint state
                    logger.warning(
                        "New step '%s' with no dependencies added to workflow — it will run from scratch on resume",
                        step.id,
                    )

        return None

    async def get_execution_history(self, workflow_name: str) -> List[Dict[str, Any]]:
        """List all executions of a workflow from the executions table."""
        if not self._db_path:
            return []
        return WorkflowStore.list_executions(self._db_path, workflow_name)

    async def get_execution_status(self, workflow_name: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get step-by-step status of a specific execution."""
        workflow = self.store.load_workflow(workflow_name)
        if not workflow:
            return None

        checkpointer = await self._get_checkpointer()
        graph = self.compiler.compile(
            workflow, checkpointer=checkpointer, store=self.store, base_dir=str(self.store.workflows_dir)
        )

        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await graph.aget_state(config)
        except Exception as e:
            logger.error("Could not get state for thread '%s': %s", thread_id, e)
            return None

        if not snapshot or not snapshot.values:
            return None

        steps_data = snapshot.values.get("steps", {})
        all_step_ids = [s.id for s in workflow.steps]

        steps_status = []
        for sid in all_step_ids:
            if sid in steps_data:
                info = steps_data[sid]
                steps_status.append(
                    {
                        "id": sid,
                        "status": info.get("status", "unknown"),
                        "execution_time": info.get("execution_time"),
                        "has_output": "result" in info.get("outputs", {}),
                        "error": info.get("outputs", {}).get("error"),
                    }
                )
            else:
                steps_status.append({"id": sid, "status": "pending"})

        return {
            "thread_id": thread_id,
            "workflow_name": workflow_name,
            "next_steps": list(snapshot.next),
            "has_interrupts": bool(snapshot.interrupts),
            "steps": steps_status,
        }

    async def resume_from_step(
        self,
        workflow_name: str,
        thread_id: str,
        step_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Resume a workflow execution by replaying from a specific step forward.

        Finds the checkpoint right before step_id ran (where step_id is in
        snapshot.next), then invokes the graph from that checkpoint.
        """
        workflow = self.store.load_workflow(workflow_name)
        if not workflow:
            return None

        step_ids = {s.id for s in workflow.steps}
        if step_id not in step_ids:
            logger.error("Step '%s' not found in workflow '%s'", step_id, workflow_name)
            return None

        checkpointer = await self._get_checkpointer()
        graph = self.compiler.compile(
            workflow, checkpointer=checkpointer, store=self.store, base_dir=str(self.store.workflows_dir)
        )

        config = {"configurable": {"thread_id": thread_id}}
        error = await self._validate_checkpoint_compatibility(graph, workflow, config)
        if error:
            return {"error": error, "steps": {}}

        target_checkpoint_id = None
        async for snapshot in graph.aget_state_history(config):
            if step_id in snapshot.next:
                target_checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")
                break

        if not target_checkpoint_id:
            logger.error("No checkpoint found before step '%s' in thread '%s'", step_id, thread_id)
            return None

        resume_config = self._make_config(thread_id)
        resume_config["configurable"]["checkpoint_id"] = target_checkpoint_id

        logger.info("Resuming from step '%s' (checkpoint: %s)", step_id, target_checkpoint_id)
        result = await graph.ainvoke(None, resume_config)
        return result
