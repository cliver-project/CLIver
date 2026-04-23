"""Gateway — the long-running daemon process for CLIver.

Hosts a cron scheduler for background tasks, an aiohttp web application
for the API, and platform adapters for messaging integrations.
"""

import asyncio
import fcntl
import importlib
import logging
import os
import tempfile
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import List, Optional

from cliver.agent_profile import CliverProfile
from cliver.config import ConfigManager
from cliver.gateway.adapter_manager import AdapterManager
from cliver.gateway.adapters import BUILTIN_ADAPTERS
from cliver.gateway.api_server import register_routes
from cliver.gateway.platform_adapter import (
    MediaAttachment,
    MessageEvent,
    PlatformAdapter,
    split_message,
)
from cliver.gateway.scheduler import CronScheduler
from cliver.gateway.task_run_store import TaskRunStore
from cliver.llm import AgentCore
from cliver.session_manager import SessionManager
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun

logger = logging.getLogger(__name__)

im_context: ContextVar[Optional[dict]] = ContextVar("im_context", default=None)


class ThreadQueue:
    """Per-thread asyncio locks for sequential message processing."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_used: dict[str, float] = {}

    def get_lock(self, session_key: str) -> asyncio.Lock:
        if session_key not in self._locks:
            self._locks[session_key] = asyncio.Lock()
        self._last_used[session_key] = time.monotonic()
        return self._locks[session_key]

    def cleanup(self, max_idle_seconds: float = 3600):
        now = time.monotonic()
        stale = [k for k, t in self._last_used.items() if now - t > max_idle_seconds]
        for key in stale:
            self._locks.pop(key, None)
            self._last_used.pop(key, None)


class Gateway:
    """Top-level orchestrator for the gateway daemon.

    Manages the lifecycle of:
    - aiohttp web application (API server)
    - CronScheduler (tick-based task execution)
    - AdapterManager (platform adapter connections)
    """

    def __init__(
        self,
        config_dir: Path,
        agent_name: str = "CLIver",
        resolved_config=None,
    ):
        self.config_dir = Path(config_dir)
        self.agent_name = agent_name
        self._resolved_config = resolved_config
        self._pid_path = self.config_dir / "cliver-gateway.pid"
        self._pid_file = None
        self._task_executor: Optional[AgentCore] = None
        self._scheduler: Optional[CronScheduler] = None
        self._run_store: Optional[TaskRunStore] = None
        self._adapter_manager: Optional[AdapterManager] = None
        self._session_manager: Optional[SessionManager] = None
        self._cron_task: Optional[asyncio.Task] = None
        self._tasks_run = 0
        self._start_time = 0.0
        self._thread_queue = ThreadQueue()

    def init(self) -> None:
        """Initialize components that need to run before fork.

        Creates AgentCore and resolves all secrets. Call this in the
        parent process before fork — the child inherits the initialized
        state and never touches the keychain.
        """
        self._task_executor = self._create_task_executor()

    def _get_config_manager(self) -> "ConfigManager":
        """Get config manager, using pre-resolved config if available."""
        return ConfigManager(self.config_dir, config=self._resolved_config)

    def create_app(self):
        """Create and return the aiohttp web application."""
        from aiohttp import web

        app = web.Application()
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    def run(self) -> None:
        """Start the gateway as an aiohttp web server."""
        # Allow nested asyncio.run() calls from tools (e.g. browser_action,
        # parallel_tasks) that assume they're in a sync context.
        import nest_asyncio

        nest_asyncio.apply()

        from aiohttp import web

        config = self._get_config_manager().config
        gw_config = config.gateway
        host = gw_config.host if gw_config else "127.0.0.1"
        port = gw_config.port if gw_config else 8321
        web.run_app(self.create_app(), host=host, port=port, print=logger.info)

    async def _on_startup(self, app) -> None:
        """Lifecycle hook: acquire flock, init components, register routes."""
        logger.info(f"Gateway starting (agent: {self.agent_name})")
        self._start_time = time.monotonic()

        # Singleton flock
        self._pid_file = open(self._pid_path, "w")
        try:
            fcntl.flock(self._pid_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._pid_file.close()
            self._pid_file = None
            raise RuntimeError("Another gateway is already running") from None
        self._pid_file.write(str(os.getpid()))
        self._pid_file.flush()

        # AgentCore (may already be set if pre-initialized before fork)
        if self._task_executor:
            logger.info("Using pre-initialized AgentCore")
        else:
            try:
                self._task_executor = self._create_task_executor()
            except Exception as e:
                logger.error(f"Failed to create task executor: {e}")

        # Register API routes
        config = self._get_config_manager().config
        gw_config = config.gateway
        api_key = gw_config.api_key if gw_config else None
        if self._task_executor:
            register_routes(app, self._task_executor, self._get_status, api_key=api_key)

        # Always register /health even without executor
        from aiohttp import web

        existing_health = any(
            r.resource.canonical == "/health" for r in app.router.routes() if hasattr(r.resource, "canonical")
        )
        if not existing_health:

            async def health_fallback(request):
                return web.json_response({"status": "degraded", **self._get_status()})

            app.router.add_get("/health", health_fallback)

        # Admin portal
        try:
            from cliver.gateway.admin import register_admin_routes

            admin_user = gw_config.admin_username if gw_config else None
            admin_pass = gw_config.admin_password if gw_config else None
            admin_ctx = {
                "get_status": self._get_status_async,
                "agent_name": self.agent_name,
                "config_dir": self.config_dir,
                "gateway": self,
            }
            register_admin_routes(app, username=admin_user, password=admin_pass, context=admin_ctx)
            if admin_user and admin_pass:
                logger.info("Admin portal enabled at /admin")
            else:
                logger.info("Admin portal disabled (no credentials configured)")
        except Exception as e:
            logger.error(f"Failed to register admin routes: {e}")

        # Cron scheduler
        try:
            agent_profile = CliverProfile(self.agent_name, self.config_dir)
            agent_profile.ensure_dirs()
            task_manager = TaskManager(agent_profile.tasks_dir)
            self._run_store = TaskRunStore(agent_profile.agent_dir / "gateway.db")

            self._scheduler = CronScheduler(
                task_manager=task_manager,
                run_store=self._run_store,
                run_task_fn=self._run_task,
            )
            if self._scheduler:
                self._scheduler.validate_tasks()
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")

        # Global asyncio exception handler — catches unhandled exceptions in tasks
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._asyncio_exception_handler)

        self._cron_task = asyncio.create_task(self._cron_loop())

        # Adapters
        try:
            adapters = self._load_adapters()
            self._adapter_manager = AdapterManager(
                adapters,
                on_message=self._handle_message,
                on_reconnect=self._on_adapter_reconnect,
            )
            await self._adapter_manager.run()
        except Exception as e:
            logger.error(f"Failed to start adapters: {e}")

        # Initialize session manager (single instance for the gateway lifetime)
        try:
            agent_profile = CliverProfile(self.agent_name, self.config_dir)
            gw_sessions_dir = agent_profile.agent_dir / "gateway-sessions"
            self._session_manager = SessionManager(gw_sessions_dir)
            sc = self._get_config_manager().config.session
            deleted = self._session_manager.delete_stale_sessions(max_age_days=sc.max_age_days)
            if deleted:
                logger.info("Cleaned up %d stale sessions (>%d days)", deleted, sc.max_age_days)
            deleted = self._session_manager.delete_oldest_sessions(keep=sc.max_sessions)
            if deleted:
                logger.info("Cleaned up %d oldest sessions (kept %d)", deleted, sc.max_sessions)
        except Exception as e:
            logger.warning("Session cleanup failed: %s", e)

        logger.info("Gateway started")

    async def _on_cleanup(self, app) -> None:
        """Lifecycle hook: stop adapters, close DB, release flock."""
        logger.info("Gateway stopping")

        if self._cron_task and not self._cron_task.done():
            self._cron_task.cancel()

        if self._adapter_manager:
            try:
                await self._adapter_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping adapters: {e}")

        if self._run_store:
            try:
                self._run_store.close()
            except Exception as e:
                logger.error(f"Error closing run store: {e}")

        self._thread_queue.cleanup(max_idle_seconds=0)

        if self._pid_file:
            self._pid_file.close()
            self._pid_file = None
        if self._pid_path.exists():
            self._pid_path.unlink(missing_ok=True)

        logger.info("Gateway stopped")

    def _get_status(self) -> dict:
        """Return current gateway status for /health endpoint."""
        uptime = int(time.monotonic() - self._start_time) if self._start_time else 0
        platforms = self._adapter_manager.connected_platforms if self._adapter_manager else []
        adapter_statuses = self._adapter_manager.platform_statuses if self._adapter_manager else []
        return {
            "uptime": uptime,
            "tasks_run": self._tasks_run,
            "platforms": platforms,
            "adapters": adapter_statuses,
        }

    async def _get_status_async(self) -> dict:
        """Async wrapper for _get_status (used by admin API)."""
        return self._get_status()

    async def _cron_loop(self) -> None:
        """Background task: tick the scheduler every 60 seconds."""
        while True:
            try:
                if self._scheduler:
                    executed = await self._scheduler.tick()
                    if executed > 0:
                        self._tasks_run += executed
                        logger.info(f"Scheduler: executed {executed} task(s)")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}")
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return

    # -- Task execution ----------------------------------------------------------

    async def _run_task(self, task: TaskDefinition) -> None:
        """Execute a task — origin-aware with reply-back for IM tasks."""
        execution_id = str(uuid.uuid4())[:8]
        logger.info(f"Running task '{task.name}' (execution: {execution_id})")

        if self._run_store:
            self._run_store.set_task_state(task.name, "running")

        # Check adapter availability for IM-origin tasks
        if task.origin and task.origin.platform:
            connected = self._adapter_manager.connected_platforms if self._adapter_manager else []
            if task.origin.platform not in connected:
                logger.warning(
                    "Task '%s' suspended: adapter '%s' not connected",
                    task.name,
                    task.origin.platform,
                )
                if self._run_store:
                    self._run_store.set_task_state(
                        task.name,
                        "suspended",
                        reason=f"Adapter '{task.origin.platform}' not connected",
                    )
                return

        run_record = TaskRun(
            task_name=task.name,
            execution_id=execution_id,
            status="running",
            started_at=TaskManager.timestamp_now(),
        )

        try:
            # Load conversation history for IM-origin tasks
            conversation_history = None
            if task.origin and task.origin.session_key and self._session_manager:
                conversation_history = self._load_origin_history(task.origin.session_key)

            if task.workflow:
                inputs = dict(task.workflow_inputs or {})
                inputs["prompt"] = task.prompt
                await self.run_workflow(task.workflow, inputs=inputs)
                response_text = "Workflow completed."
            else:
                system_appender = None
                if task.skills:
                    system_appender = self._build_skill_appender(task.skills)

                response = await self._task_executor.process_user_input(
                    user_input=task.prompt,
                    model=task.model,
                    system_message_appender=system_appender,
                    conversation_history=conversation_history,
                )
                response_text = str(response.content) if response and response.content else "No response."

            run_record.status = "completed"

            # Deliver result to IM origin
            if task.origin and task.origin.platform and task.origin.channel_id:
                await self._deliver_to_origin(task, response_text)

        except Exception as e:
            run_record.status = "failed"
            run_record.error = str(e)
            response_text = f"Task '{task.name}' failed: {e}"
            logger.error(f"Task '{task.name}' failed: {e}")

            if task.origin and task.origin.platform and task.origin.channel_id:
                try:
                    await self._deliver_to_origin(task, response_text)
                except Exception:
                    logger.error("Failed to deliver error to origin")

        finally:
            run_record.finished_at = TaskManager.timestamp_now()
            if self._run_store:
                self._run_store.record_run(run_record)
                self._run_store.set_task_state(task.name, run_record.status)

    def _load_origin_history(self, session_key: str):
        """Load conversation history from an IM session for task context."""
        from langchain_core.messages import AIMessage, HumanMessage

        sessions = self._session_manager.list_sessions()
        session_id = None
        for s in sessions:
            if s.get("title") == session_key:
                session_id = s["id"]
                break

        if not session_id:
            return None

        turns = self._session_manager.load_turns(session_id)
        if not turns:
            return None

        history = []
        for turn in turns:
            if turn["role"] == "user":
                history.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                history.append(AIMessage(content=turn["content"]))
        return history or None

    async def _deliver_to_origin(self, task: TaskDefinition, response_text: str) -> None:
        """Deliver task result to the originating IM thread and append synthetic turns."""
        adapter = self._get_adapter(task.origin.platform)
        if not adapter:
            logger.error(f"No adapter for platform '{task.origin.platform}'")
            return

        formatted = adapter.format_message(response_text)
        chunks = split_message(formatted, adapter.max_message_length())
        for chunk in chunks:
            await adapter.send_text(
                task.origin.channel_id,
                chunk,
                reply_to=task.origin.thread_id,
            )

        # Append synthetic turns to the origin session
        if self._session_manager and task.origin.session_key:
            session_id = self._get_or_create_session(
                self._session_manager,
                task.origin.session_key,
            )
            self._session_manager.append_turn(
                session_id,
                "user",
                f"[Task '{task.name}' executed]",
            )
            self._session_manager.append_turn(
                session_id,
                "assistant",
                response_text,
            )

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

        config_manager = self._get_config_manager()
        executor = WorkflowExecutor(
            task_executor=self._task_executor,
            store=store,
            db_path=db_path,
            app_config=config_manager.config,
        )

        return await executor.execute_workflow(workflow_name, inputs=inputs)

    def _create_task_executor(self) -> AgentCore:
        """Create a AgentCore from config (same config the CLI uses)."""
        config_manager = self._get_config_manager()
        agent_profile = CliverProfile(self.agent_name, self.config_dir)
        agent_profile.ensure_dirs()

        tool_handler = _create_gateway_tool_handler()

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

        Each thread gets its own session. For top-level messages (no thread),
        the message's own ID becomes the thread root — the bot's reply will
        create a new thread, giving each conversation its own context.

        Key format: platform:channel_id:thread_id
        """
        thread = event.thread_id or event.message_id or ""
        return f"{event.platform}:{event.channel_id}:{thread}"

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
        config_manager = self._get_config_manager()
        gateway_config = config_manager.config.gateway
        if not gateway_config or not gateway_config.platforms:
            return []

        adapters = []
        for name, platform_config in gateway_config.platforms.items():
            try:
                if not platform_config.name:
                    platform_config.name = name
                class_path = self._resolve_adapter_class(platform_config.type)
                adapter_cls = self._import_adapter_class(class_path)
                adapter = adapter_cls(platform_config)
                adapters.append(adapter)
                logger.info(f"Loaded adapter: {adapter.name}")
            except Exception as e:
                logger.error(f"Failed to load adapter '{name}': {e}")

        return adapters

    # -- Message handling -----------------------------------------------------

    async def _compress_history(self, history, new_input: str):
        """Compress conversation history if it exceeds the model's context window."""
        from cliver.conversation_compressor import ConversationCompressor

        llm_engine = self._task_executor.get_llm_engine()
        model_config = self._task_executor._get_llm_model()
        context_window = getattr(model_config, "context_window", None) or 32768

        compressor = ConversationCompressor(context_window=context_window)
        if compressor.needs_compression([], history, new_input):
            logger.info("Compressing conversation history (%d turns)", len(history))
            history = await compressor.compress(history, llm_engine)
            logger.info("Compressed to %d turns", len(history))
        return history

    def _asyncio_exception_handler(self, loop, context):
        exception = context.get("exception")
        message = context.get("message", "")
        logger.error("Asyncio unhandled exception: %s (message: %s)", exception, message)
        if exception:
            import traceback

            logger.error("".join(traceback.format_exception(type(exception), exception, exception.__traceback__)))

    async def _handle_message(self, event: MessageEvent) -> None:
        """Central message handler -- routes platform messages to AgentCore."""
        try:
            await self._handle_message_inner(event)
        except Exception as e:
            logger.exception("_handle_message crashed: %s", e)

    async def _handle_message_inner(self, event: MessageEvent) -> None:
        logger.info(
            "_handle_message called: platform=%s, user=%s, text=%.100s", event.platform, event.user_id, event.text
        )
        if not self._task_executor:
            logger.error("AgentCore not initialized — cannot process message")
            return

        # Find the adapter for this platform
        adapter = self._get_adapter(event.platform)
        if not adapter:
            logger.error(f"No adapter found for platform '{event.platform}'")
            return

        # Resolve session — uses the shared SessionManager instance
        session_key = self._resolve_session_key(event)
        async with self._thread_queue.get_lock(session_key):
            sm = self._session_manager

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

            # Compress history if it exceeds the model's context window
            if conversation_history and self._task_executor:
                try:
                    conversation_history = await self._compress_history(conversation_history, event.text)
                except Exception as e:
                    logger.warning("History compression failed, using full history: %s", e)

            # Record user turn
            sm.append_turn(session_id, "user", event.text)

            # Run AgentCore
            logger.info(
                "Calling AgentCore.process_user_input (model=%s, history=%d turns)",
                self._task_executor.default_model,
                len(conversation_history),
            )
            # Set IM context for tools (e.g., create_task) to read
            im_context.set(
                {
                    "platform": event.platform,
                    "channel_id": event.channel_id,
                    "thread_id": event.thread_id or event.message_id,
                    "user_id": event.user_id,
                    "session_key": session_key,
                }
            )
            try:
                response = await self._task_executor.process_user_input(
                    user_input=event.text,
                    images=images or None,
                    audio_files=audio_files or None,
                    conversation_history=conversation_history or None,
                )

                from cliver.media_handler import MultimediaResponseHandler

                handler = MultimediaResponseHandler()
                multimedia = handler.process_response(response)
                response_text = multimedia.text_content or "No response."
                logger.info("AgentCore response: %.200s", response_text)
            except Exception as e:
                logger.exception("AgentCore error: %s", e)
                response_text = f"Error: {e}"
                multimedia = None
            finally:
                im_context.set(None)

            # Record assistant turn and trim if session is getting large
            sm.append_turn(session_id, "assistant", response_text)
            sc = self._get_config_manager().config.session
            trimmed = sm.trim_turns(session_id, keep_last=sc.max_turns_per_session)
            if trimmed:
                logger.info("Trimmed %d old turns from session %s", trimmed, session_id)

            # Reply in-thread: use existing thread_id, or message_id to create a new thread
            reply_to = event.thread_id or event.message_id

            # Send text response
            formatted = adapter.format_message(response_text)
            chunks = split_message(formatted, adapter.max_message_length())
            logger.info(
                "Sending %d chunk(s) to %s channel %s (thread: %s)",
                len(chunks),
                event.platform,
                event.channel_id,
                reply_to,
            )

            for chunk in chunks:
                try:
                    await adapter.send_text(event.channel_id, chunk, reply_to=reply_to)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")

            # Send media (images, files, audio)
            if multimedia and multimedia.has_media():
                for media in multimedia.media_content:
                    try:
                        if media.data and media.type.value == "image":
                            await adapter.send_image(event.channel_id, media.data, caption=media.filename or "")
                        elif media.data and media.type.value == "audio":
                            await adapter.send_voice(event.channel_id, media.data)
                        elif media.data:
                            await adapter.send_file(event.channel_id, media.data, filename=media.filename or "file")
                        logger.info("Sent %s media to %s", media.type.value, event.channel_id)
                    except Exception as e:
                        logger.error("Failed to send %s media: %s", media.type.value, e)

    def _get_adapter(self, platform_name: str) -> Optional[PlatformAdapter]:
        """Find the adapter for a given platform name."""
        if not self._adapter_manager:
            return None
        for adapter in self._adapter_manager._adapters:
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

    async def _on_adapter_reconnect(self, platform_name: str) -> None:
        """Resume suspended tasks when an adapter reconnects."""
        if not self._run_store:
            return

        agent_profile = CliverProfile(self.agent_name, self.config_dir)
        task_manager = TaskManager(agent_profile.tasks_dir)

        suspended = self._run_store.get_tasks_by_status("suspended")
        for state in suspended:
            task = task_manager.get_task(state["task_name"])
            if task and task.origin and task.origin.platform == platform_name:
                self._run_store.set_task_state(task.name, "pending")
                logger.info(
                    "Resumed suspended task '%s' (adapter '%s' reconnected)",
                    task.name,
                    platform_name,
                )


def _create_gateway_tool_handler():
    """Create a tool event handler that logs via the standard logger."""
    from cliver.tool_events import ToolEvent, ToolEventType

    tool_logger = logging.getLogger("cliver.gateway.tools")

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
