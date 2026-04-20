"""Gateway — the long-running daemon process for CLIver.

Hosts a cron scheduler for background tasks, a control endpoint
for status/stop, and platform adapters for messaging integrations.
"""

import asyncio
import importlib
import logging
import logging.handlers
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from cliver.agent_profile import CliverProfile
from cliver.config import ConfigManager
from cliver.gateway.adapters import BUILTIN_ADAPTERS
from cliver.gateway.control import ControlServer
from cliver.gateway.platform_adapter import (
    MediaAttachment,
    MessageEvent,
    PlatformAdapter,
    split_message,
)
from cliver.gateway.scheduler import CronScheduler
from cliver.llm import AgentCore
from cliver.session_manager import SessionManager
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun

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
        self._task_executor: Optional[AgentCore] = None
        self._api_server = None

    async def start(self) -> None:
        """Initialize components and start the control server."""
        logger.info(f"Gateway starting (agent: {self.agent_name})")

        # Initialize AgentCore
        self._task_executor = self._create_task_executor()

        # Initialize cron scheduler
        agent_profile = CliverProfile(self.agent_name, self.config_dir)
        agent_profile.ensure_dirs()
        task_manager = TaskManager(agent_profile.tasks_dir)
        cron_state_path = agent_profile.agent_dir / "cron-state.json"

        self._scheduler = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=self._run_task,
        )

        # Load and start platform adapters
        self._adapters = self._load_adapters()
        for adapter in self._adapters:
            try:
                await adapter.start(on_message=self._handle_message)
                logger.info(f"Started adapter: {adapter.name}")
            except Exception as e:
                logger.error(f"Failed to start adapter {adapter.name}: {e}")

        # Start API server if configured
        config_manager = ConfigManager(self.config_dir)
        gateway_config = config_manager.config.gateway
        if gateway_config and gateway_config.api_server and gateway_config.api_server.enabled:
            try:
                from cliver.gateway.api_server import APIServer

                self._api_server = APIServer(
                    task_executor=self._task_executor,
                    config=gateway_config.api_server,
                )
                await self._api_server.start()
            except ImportError:
                logger.error("aiohttp not installed. Install with: pip install cliver[api]")
            except Exception as e:
                logger.error(f"Failed to start API server: {e}")

        # Start control server
        await self._control_server.start()
        self.is_running = True
        logger.info("Gateway started")

    async def stop(self) -> None:
        """Stop all components and clean up."""
        logger.info("Gateway stopping")

        # Stop API server
        if self._api_server:
            try:
                await self._api_server.stop()
            except Exception as e:
                logger.error(f"Error stopping API server: {e}")

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
        """Execute a scheduled task — either as a workflow or a chat prompt."""
        execution_id = str(uuid.uuid4())[:8]
        logger.info(f"Running task '{task.name}' (execution: {execution_id})")

        agent_profile = CliverProfile(self.agent_name, self.config_dir)
        task_manager = TaskManager(agent_profile.tasks_dir)

        run_record = TaskRun(
            task_name=task.name,
            execution_id=execution_id,
            status="running",
            started_at=TaskManager.timestamp_now(),
        )

        try:
            if task.workflow:
                # Run as workflow — prompt is injected as inputs.prompt
                inputs = dict(task.workflow_inputs or {})
                inputs["prompt"] = task.prompt
                await self.run_workflow(task.workflow, inputs=inputs)
            else:
                # Run as chat prompt with optional skill pre-activation
                system_appender = None
                if task.skills:
                    system_appender = self._build_skill_appender(task.skills)

                await self._task_executor.process_user_input(
                    user_input=task.prompt,
                    model=task.model,
                    system_message_appender=system_appender,
                )
            run_record.status = "completed"
        except Exception as e:
            run_record.status = "failed"
            run_record.error = str(e)
            logger.error(f"Task '{task.name}' failed: {e}")
        finally:
            run_record.finished_at = TaskManager.timestamp_now()
            task_manager.record_run(run_record)

    def _build_skill_appender(self, skill_names: list) -> callable:
        """Build a system_message_appender that injects pre-activated skills."""
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        skill_contents = []
        for name in skill_names:
            skill = manager.get_skill(name)
            if skill and skill.body:
                skill_contents.append(f"# Skill: {skill.name}\n\n{skill.body}")
            else:
                logger.warning("Skill '%s' not found for pre-activation", name)

        if not skill_contents:
            return None

        combined = "\n\n".join(skill_contents)
        return lambda: combined

    async def run_workflow(self, workflow_name: str, inputs: dict = None) -> Optional[dict]:
        """Execute a workflow headlessly via the gateway."""
        from cliver.workflow.persistence import WorkflowStore
        from cliver.workflow.workflow_executor import WorkflowExecutor

        if not self._task_executor:
            logger.error("Gateway not started — cannot run workflow")
            return None

        agent_profile = CliverProfile(self.agent_name, self.config_dir)
        store = WorkflowStore(agent_profile.workflows_dir)
        db_path = agent_profile.agent_dir / "workflow-checkpoints.db"

        config_manager = ConfigManager(self.config_dir)
        executor = WorkflowExecutor(
            task_executor=self._task_executor,
            store=store,
            db_path=db_path,
            app_config=config_manager.config,
        )

        return await executor.execute_workflow(workflow_name, inputs=inputs)

    def _create_task_executor(self) -> AgentCore:
        """Create a AgentCore from config (same config the CLI uses)."""
        config_manager = ConfigManager(self.config_dir)
        agent_profile = CliverProfile(self.agent_name, self.config_dir)
        agent_profile.ensure_dirs()

        gateway_config = config_manager.config.gateway
        tool_handler = _create_gateway_tool_handler(self.config_dir, gateway_config)

        executor = AgentCore(
            llm_models=config_manager.list_llm_models(),
            mcp_servers=config_manager.list_mcp_servers_for_mcp_caller(),
            default_model=(config_manager.get_llm_model().name if config_manager.get_llm_model() else None),
            user_agent=config_manager.config.user_agent,
            agent_name=self.agent_name,
            agent_profile=agent_profile,
            enabled_toolsets=config_manager.config.enabled_toolsets,
            on_tool_event=tool_handler,
        )
        executor.configure_rate_limits(config_manager.config.providers)
        return executor

    # -- Session & adapter resolution -----------------------------------------

    @staticmethod
    def _resolve_session_key(event: MessageEvent) -> str:
        """Build a session key from a message event.

        Groups: keyed by platform:channel_id (shared context).
        DMs: keyed by platform:user_id (private context).
        """
        if event.is_group:
            return f"{event.platform}:{event.channel_id}"
        return f"{event.platform}:{event.user_id}"

    @staticmethod
    def _resolve_adapter_class(adapter_type: str) -> str:
        """Resolve an adapter type to its full module.ClassName path.

        Builtin types (no dots) are looked up in BUILTIN_ADAPTERS.
        Custom types (with dots) are returned as-is.
        """
        if "." not in adapter_type:
            if adapter_type not in BUILTIN_ADAPTERS:
                raise ValueError(
                    f"Unknown adapter type '{adapter_type}'. Available: {', '.join(BUILTIN_ADAPTERS.keys())}"
                )
            return BUILTIN_ADAPTERS[adapter_type]
        return adapter_type

    @staticmethod
    def _import_adapter_class(class_path: str):
        """Dynamically import and return an adapter class."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _load_adapters(self) -> List[PlatformAdapter]:
        """Load and instantiate adapters from config."""
        config_manager = ConfigManager(self.config_dir)
        gateway_config = config_manager.config.gateway
        if not gateway_config or not gateway_config.platforms:
            return []

        adapters = []
        for platform_config in gateway_config.platforms:
            try:
                class_path = self._resolve_adapter_class(platform_config.type)
                adapter_cls = self._import_adapter_class(class_path)
                adapter = adapter_cls(platform_config)
                adapters.append(adapter)
                logger.info(f"Loaded adapter: {adapter.name}")
            except Exception as e:
                logger.error(f"Failed to load adapter '{platform_config.type}': {e}")

        return adapters

    # -- Message handling -----------------------------------------------------

    async def _handle_message(self, event: MessageEvent) -> None:
        """Central message handler -- routes platform messages to AgentCore."""
        if not self._task_executor:
            logger.error("AgentCore not initialized")
            return

        # Find the adapter for this platform
        adapter = self._get_adapter(event.platform)
        if not adapter:
            logger.error(f"No adapter found for platform '{event.platform}'")
            return

        # Resolve session
        session_key = self._resolve_session_key(event)
        agent_profile = CliverProfile(self.agent_name, self.config_dir)
        sm = SessionManager(agent_profile.sessions_dir)

        # Get or create session
        session_id = self._get_or_create_session(sm, session_key)

        # Show typing indicator
        try:
            await adapter.send_typing(event.channel_id)
        except Exception as e:
            logger.debug(f"Typing indicator failed: {e}")

        # Prepare media as temp files for AgentCore
        images, audio_files = [], []
        transcribed_texts = []
        for media in event.media:
            if media.data:
                suffix = _media_suffix(media)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(media.data)
                tmp.close()
                if media.type == "image":
                    images.append(tmp.name)
                elif media.type == "voice":
                    # Auto-transcribe voice messages to text
                    try:
                        from cliver.tools.transcribe_audio import transcribe_voice_message

                        transcript = await transcribe_voice_message(tmp.name)
                        if transcript:
                            transcribed_texts.append(transcript)
                            logger.info("Voice message transcribed: %.100s", transcript)
                        else:
                            audio_files.append(tmp.name)
                    except Exception as e:
                        logger.debug(f"Voice transcription failed, passing raw audio: {e}")
                        audio_files.append(tmp.name)

        # Prepend transcribed voice text to user input
        if transcribed_texts:
            voice_text = " ".join(transcribed_texts)
            if event.text:
                event.text = f"{event.text}\n\n[Voice message]: {voice_text}"
            else:
                event.text = voice_text

        # Load conversation history
        turns = sm.load_turns(session_id)
        from langchain_core.messages import AIMessage, HumanMessage

        conversation_history = []
        for turn in turns:
            if turn["role"] == "user":
                conversation_history.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                conversation_history.append(AIMessage(content=turn["content"]))

        # Record user turn
        sm.append_turn(session_id, "user", event.text)

        # Run AgentCore
        try:
            response = await self._task_executor.process_user_input(
                user_input=event.text,
                images=images or None,
                audio_files=audio_files or None,
                conversation_history=conversation_history or None,
            )

            response_text = str(response.content) if response and response.content else "No response."
        except Exception as e:
            logger.error(f"AgentCore error: {e}")
            response_text = f"Error: {e}"

        # Record assistant turn
        sm.append_turn(session_id, "assistant", response_text)

        # Format and send response
        formatted = adapter.format_message(response_text)
        chunks = split_message(formatted, adapter.max_message_length())

        for chunk in chunks:
            try:
                await adapter.send_text(event.channel_id, chunk)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")

    def _get_adapter(self, platform_name: str) -> Optional[PlatformAdapter]:
        """Find the adapter for a given platform name."""
        for adapter in self._adapters:
            if adapter.name == platform_name:
                return adapter
        return None

    def _get_or_create_session(self, sm: SessionManager, session_key: str) -> str:
        """Get existing session by key or create a new one.

        Uses the session title field to store the session key for lookup.
        """
        sessions = sm.list_sessions()
        for s in sessions:
            if s.get("title") == session_key:
                return s["id"]
        return sm.create_session(title=session_key)


def _create_gateway_tool_handler(config_dir: Path, gateway_config=None):
    """Create a tool event handler that logs to a rotating file.

    Returns None if no gateway config is provided (tool events are silent).
    """
    from cliver.tool_events import ToolEvent, ToolEventType

    # Resolve log file path
    if gateway_config and gateway_config.log_file:
        log_path = Path(gateway_config.log_file)
    else:
        log_path = Path(config_dir) / "gateway.log"

    max_bytes = gateway_config.log_max_bytes if gateway_config else 10 * 1024 * 1024
    backup_count = gateway_config.log_backup_count if gateway_config else 5

    # Set up a dedicated rotating logger
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tool_logger = logging.getLogger("cliver.gateway.tools")
    tool_logger.setLevel(logging.INFO)
    # Avoid duplicate handlers on restart
    if not tool_logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        tool_logger.addHandler(handler)

    def _handler(event: ToolEvent) -> None:
        if event.event_type == ToolEventType.TOOL_START:
            args_summary = ""
            if event.args:
                parts = []
                for k, v in event.args.items():
                    val = str(v)
                    if len(val) > 200:
                        val = val[:197] + "…"
                    parts.append(f"{k}={val}")
                args_summary = " " + ", ".join(parts[:5])
            tool_logger.info("[START] %s%s", event.tool_name, args_summary)

        elif event.event_type == ToolEventType.TOOL_END:
            duration = f" ({event.duration_ms:.0f}ms)" if event.duration_ms else ""
            tool_logger.info("[DONE]  %s%s", event.tool_name, duration)
            if event.result:
                # Log full result (the rotating handler limits total disk usage)
                for line in event.result.splitlines():
                    tool_logger.info("        %s", line)

        elif event.event_type == ToolEventType.TOOL_ERROR:
            duration = f" ({event.duration_ms:.0f}ms)" if event.duration_ms else ""
            tool_logger.warning("[ERROR] %s%s: %s", event.tool_name, duration, event.error)

    return _handler


def _media_suffix(media: MediaAttachment) -> str:
    """Determine file suffix from media type/mime."""
    if media.mime_type:
        ext = media.mime_type.split("/")[-1]
        return f".{ext}"
    if media.type == "image":
        return ".png"
    if media.type == "voice":
        return ".ogg"
    return ".bin"
