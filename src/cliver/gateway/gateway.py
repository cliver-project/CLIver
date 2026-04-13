"""Gateway — the long-running daemon process for CLIver.

Hosts a cron scheduler for background tasks, a control endpoint
for status/stop, and platform adapters for messaging integrations.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from cliver.agent_profile import AgentProfile
from cliver.config import ConfigManager
from cliver.gateway.control import ControlServer
from cliver.gateway.platform_adapter import PlatformAdapter
from cliver.gateway.scheduler import CronScheduler
from cliver.llm import TaskExecutor
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager

logger = logging.getLogger(__name__)


class Gateway:
    """Top-level orchestrator for the gateway daemon.

    Manages the lifecycle of:
    - ControlServer (Unix socket for status/stop)
    - CronScheduler (tick-based task execution)
    - PlatformAdapters (future messaging integrations)
    """

    def __init__(
        self,
        config_dir: Path,
        agent_name: str = "CLIver",
    ):
        self.config_dir = Path(config_dir)
        self.agent_name = agent_name
        self.is_running = False

        self._control_server = ControlServer(
            socket_path=self.config_dir / "cliver-gateway.sock",
            pid_path=self.config_dir / "cliver-gateway.pid",
        )
        self._scheduler: Optional[CronScheduler] = None
        self._adapters: List[PlatformAdapter] = []
        self._task_executor: Optional[TaskExecutor] = None
        self._workflow_executor: Optional[WorkflowExecutor] = None

    async def start(self) -> None:
        """Initialize components and start the control server."""
        logger.info(f"Gateway starting (agent: {self.agent_name})")

        # Initialize TaskExecutor and WorkflowExecutor
        self._task_executor = self._create_task_executor()
        self._workflow_executor = self._create_workflow_executor()

        # Initialize cron scheduler
        agent_profile = AgentProfile(self.agent_name, self.config_dir)
        agent_profile.ensure_dirs()
        task_manager = TaskManager(agent_profile.tasks_dir)
        cron_state_path = agent_profile.agent_dir / "cron-state.json"

        self._scheduler = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=self._run_task,
        )

        # Start control server
        await self._control_server.start()
        self.is_running = True
        logger.info("Gateway started")

    async def stop(self) -> None:
        """Stop all components and clean up."""
        logger.info("Gateway stopping")

        # Stop adapters
        for adapter in self._adapters:
            try:
                await adapter.stop()
            except Exception as e:
                logger.error(f"Error stopping adapter {adapter.name}: {e}")

        # Stop control server
        await self._control_server.stop()
        self.is_running = False
        logger.info("Gateway stopped")

    async def run(self, tick_interval: float = 60.0) -> None:
        """Run the main loop until shutdown is requested.

        Args:
            tick_interval: Seconds between scheduler ticks (default 60).
        """
        logger.info(f"Gateway main loop started (tick every {tick_interval}s)")

        while not self._control_server.shutdown_requested:
            # Run a scheduler tick
            try:
                executed = await self._scheduler.tick()
                if executed > 0:
                    self._control_server.tasks_run += executed
                    logger.info(f"Scheduler tick: executed {executed} task(s)")
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}")

            # Sleep in small increments so we can respond to shutdown quickly
            elapsed = 0.0
            while elapsed < tick_interval:
                if self._control_server.shutdown_requested:
                    break
                await asyncio.sleep(min(0.5, tick_interval - elapsed))
                elapsed += 0.5

        logger.info("Gateway main loop exiting (shutdown requested)")

    async def _run_task(self, task: TaskDefinition) -> None:
        """Execute a scheduled task via the workflow executor."""
        execution_id = str(uuid.uuid4())[:8]
        logger.info(f"Running task '{task.name}' (execution: {execution_id})")

        agent_profile = AgentProfile(self.agent_name, self.config_dir)
        task_manager = TaskManager(agent_profile.tasks_dir)

        run_record = TaskRun(
            task_name=task.name,
            execution_id=execution_id,
            workflow_name=task.workflow,
            status="running",
            started_at=TaskManager.timestamp_now(),
        )

        try:
            await self._workflow_executor.execute_workflow(
                workflow_name=task.workflow,
                inputs=task.inputs or {},
                execution_id=execution_id,
            )
            run_record.status = "completed"
        except Exception as e:
            run_record.status = "failed"
            run_record.error = str(e)
            logger.error(f"Task '{task.name}' failed: {e}")
        finally:
            run_record.finished_at = TaskManager.timestamp_now()
            task_manager.record_run(run_record)

    def _create_task_executor(self) -> TaskExecutor:
        """Create a TaskExecutor from config (same config the CLI uses)."""
        config_manager = ConfigManager(self.config_dir)
        agent_profile = AgentProfile(self.agent_name, self.config_dir)
        agent_profile.ensure_dirs()

        return TaskExecutor(
            llm_models=config_manager.list_llm_models(),
            mcp_servers=config_manager.list_mcp_servers_for_mcp_caller(),
            default_model=(config_manager.get_llm_model().name if config_manager.get_llm_model() else None),
            user_agent=config_manager.config.user_agent,
            agent_name=self.agent_name,
            agent_profile=agent_profile,
        )

    def _create_workflow_executor(self) -> WorkflowExecutor:
        """Create a WorkflowExecutor from config."""
        config_manager = ConfigManager(self.config_dir)
        workflow_config = config_manager.config.workflow
        workflow_dirs = workflow_config.workflow_dirs if workflow_config else None

        return WorkflowExecutor(
            task_executor=self._task_executor,
            workflow_manager=LocalDirectoryWorkflowManager(workflow_dirs),
        )
