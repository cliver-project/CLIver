import asyncio
import logging
import random
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.deepseek_engine import DeepSeekInferenceEngine
from cliver.llm.errors import TaskTimeoutError, get_friendly_error_message
from cliver.llm.llm_utils import is_thinking, normalize_tool_calls
from cliver.llm.media_generation import get_image_helper
from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
from cliver.llm.rate_limiter import RateLimiter, parse_period
from cliver.mcp_server_caller import MCPServersCaller
from cliver.media import load_media_file
from cliver.model_capabilities import ModelCapability
from cliver.permissions import PermissionAction, PermissionDecision, PermissionManager
from cliver.prompt_enhancer import apply_template
from cliver.tool_events import ToolEvent, ToolEventHandler, ToolEventType
from cliver.tool_registry import ToolRegistry
from cliver.util import read_context_files, retry_with_confirmation_async

# Create a logger for this module
logger = logging.getLogger(__name__)

# Maximum consecutive tool error iterations before stopping the Re-Act loop.
# This prevents infinite loops where the LLM keeps calling the same broken tool.
MAX_CONSECUTIVE_ERRORS = 3


def create_llm_engine(
    model: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"
) -> Optional[LLMInferenceEngine]:
    provider_type = model.get_provider_type()
    if provider_type == "ollama":
        return OllamaLlamaInferenceEngine(model, user_agent=user_agent, agent_name=agent_name)
    elif provider_type == "openai":
        # Route to model-specific engines for providers with API quirks
        name = (model.name_in_provider or model.name).lower()
        if name.startswith("deepseek"):
            return DeepSeekInferenceEngine(model, user_agent=user_agent, agent_name=agent_name)
        return OpenAICompatibleInferenceEngine(model, user_agent=user_agent, agent_name=agent_name)
    elif provider_type == "anthropic":
        from cliver.llm.anthropic_engine import AnthropicInferenceEngine

        return AnthropicInferenceEngine(model, user_agent=user_agent, agent_name=agent_name)
    return None


## TODO: we need to improve this to take consideration of user_input and even call some mcp tools
## TODO: We may need to parsed the structured context file and do embedding to get only related sections.
async def default_enhance_prompt(user_input: str, mcp_caller: MCPServersCaller) -> list[BaseMessage]:
    """
    Default enhancement function that reads context files for context.
    By default, it looks for Cliver.md but can be configured to look for other files.

    Args:
        user_input: The user's input
        mcp_caller: The MCP servers caller instance

    Returns:
        A list of BaseMessage with the context information
    """
    import os

    context = read_context_files(os.getcwd())
    if context:
        return [SystemMessage(content=f"Context information:\n{context}")]
    return []


# Tool registry — configured lazily by AgentCore with enabled_toolsets from config
tool_registry = ToolRegistry()


def _extract_last_content(messages: List[BaseMessage]) -> str | None:
    """Extract the last AI content from messages for partial timeout results."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, list):
                texts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return "\n".join(texts) if texts else None
            return str(content)
    return None


def _sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Sanitize messages before sending to the LLM API.

    Removes surrogate characters that can cause API errors.
    """
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Remove surrogate characters
            msg.content = msg.content.encode("utf-8", errors="replace").decode("utf-8")
    return messages


class AgentCore:
    """
    This is the central place managing the execution of all configured LLM models and MCP servers.
    """

    def __init__(
        self,
        llm_models: Dict[str, ModelConfig],
        mcp_servers: Dict[str, Dict],
        default_model: Optional[str] = None,
        user_agent: Optional[str] = None,
        agent_name: str = "CLIver",
        on_tool_event: Optional[ToolEventHandler] = None,
        agent_profile=None,
        token_tracker=None,
        permission_manager: Optional[PermissionManager] = None,
        on_permission_prompt: Optional[Callable[[str, dict], str]] = None,
        enabled_toolsets: Optional[List[str]] = None,
    ):
        self.llm_models = llm_models
        self.default_model = default_model
        self.user_agent = user_agent
        self.agent_name = agent_name
        self.on_tool_event = on_tool_event
        self.agent_profile = agent_profile  # CliverProfile for memory/identity
        self.token_tracker = token_tracker  # TokenTracker for usage auditing
        self.permission_manager = permission_manager
        self.on_permission_prompt = on_permission_prompt
        self.mcp_caller = MCPServersCaller(mcp_servers=mcp_servers)
        self.llm_engines: Dict[str, LLMInferenceEngine] = {}

        # Configure tool registry with toolsets from config
        if enabled_toolsets is not None:
            global tool_registry
            tool_registry = ToolRegistry(enabled_toolsets=enabled_toolsets)

        # Session-scoped model exclusion — prevents excluded models from
        # being used as fallback targets.  Managed via /session option commands.
        self.excluded_models: set[str] = set()

        # Tool call counter for skill auto-learning (reset per process_user_input call)
        self._tool_call_count = 0

        # Accumulated media from tool calls (e.g., ImageGenerate).
        # Reset per process_user_input call, attached to final response.
        self._generated_media: list = []

        # Per-provider rate limiter (configured via configure_rate_limits)
        self.rate_limiter = RateLimiter()

        # Set the active profile and executor so builtin tools can access them
        if agent_profile:
            from cliver.agent_profile import set_current_profile

            set_current_profile(agent_profile)

        from cliver.agent_profile import set_task_executor

        set_task_executor(self)

    def _emit_tool_event(self, event: ToolEvent) -> None:
        """Emit a tool event to the registered handler, if any."""
        if self.on_tool_event:
            try:
                self.on_tool_event(event)
            except Exception as e:
                logger.debug(f"Tool event handler error: {e}")

    def configure_rate_limits(self, providers: dict) -> None:
        """Configure rate limiter from provider configs."""
        for name, prov_config in providers.items():
            rate_limit = getattr(prov_config, "rate_limit", None)
            if rate_limit:
                try:
                    period_s = parse_period(rate_limit.period)
                    self.rate_limiter.configure(name, rate_limit.requests, period_s, rate_limit.margin)
                except Exception as e:
                    logger.warning("Invalid rate limit for provider %s: %s", name, e)

    async def _wait_for_rate_limit(self, model: str) -> None:
        """Wait for rate limit if the model's provider has one configured."""
        model_config = self._get_llm_model(model)
        if not model_config:
            return
        provider_name = model_config.provider
        wait_time = self.rate_limiter.get_wait_time(provider_name)
        if wait_time > 0.01:
            self._emit_tool_event(
                ToolEvent(
                    event_type=ToolEventType.MODEL_RATE_LIMIT,
                    tool_name=model,
                    result=f"waiting {wait_time:.1f}s",
                )
            )
        await self.rate_limiter.wait(provider_name)

    async def generate_image(self, prompt: str, model: str = None, **params) -> BaseMessage:
        """Generate images from a text prompt via the provider's image API.

        All HTTP calls are made here — helpers only format requests and parse
        responses. This keeps all external API calls centralized in AgentCore.

        Resolution order:
        1. If model specified → use its provider's image_url
        2. Scan llm_models for TEXT_TO_IMAGE capability
        3. Scan providers for any with image_url set
        """
        provider_config, model_name = self._resolve_image_provider(model)
        if not provider_config or not getattr(provider_config, "image_url", None):
            return AIMessage(content="Error: No model or provider with image generation support configured.")

        await self.rate_limiter.wait(provider_config.name)

        try:
            helper = get_image_helper(provider_config.image_url)
            request_body = helper.build_request(prompt, model_name, **params)
            response_data = await self._call_generation_api(
                provider_config.image_url,
                provider_config.get_api_key(),
                request_body,
            )
            media_list = helper.parse_response(response_data)
        except Exception as e:
            logger.error("Image generation failed: %s", e)
            return AIMessage(content=f"Error generating image: {e}")

        # Accumulate media so it can be attached to the final Re-Act response
        self._generated_media.extend(media_list)

        urls = [m.data for m in media_list if m.data]
        content = f"Generated {len(media_list)} image(s):\n" + "\n".join(urls) if urls else "Image generated."

        return AIMessage(
            content=content,
            additional_kwargs={"media_content": media_list},
        )

    async def _call_generation_api(self, url: str, api_key: str, body: dict) -> dict:
        """Make an HTTP POST to a generation API endpoint.

        Centralizes all generation API calls (image, audio) so that
        rate limiting, error handling, and logging are in one place.
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def _resolve_image_provider(self, model: str = None):
        """Resolve which provider and model name to use for image generation.

        Returns (ProviderConfig, model_name) or (None, None).
        """
        from cliver.model_capabilities import ModelCapability

        # 1. Explicit model
        if model:
            mc = self._get_llm_model(model)
            if mc:
                prov = getattr(mc, "_provider_config", None)
                if prov and getattr(prov, "image_url", None):
                    return prov, mc.name_in_provider or mc.name

        # 2. Scan for model with TEXT_TO_IMAGE capability
        for _name, mc in self.llm_models.items():
            caps = mc.get_capabilities()
            if ModelCapability.TEXT_TO_IMAGE in caps:
                prov = getattr(mc, "_provider_config", None)
                if prov and getattr(prov, "image_url", None):
                    return prov, mc.name_in_provider or mc.name

        # 3. Scan all models' providers for any with image_url
        for _name, mc in self.llm_models.items():
            prov = getattr(mc, "_provider_config", None)
            if prov and getattr(prov, "image_url", None):
                return prov, None

        return None, None

    def _track_tokens(self, response, model: str) -> None:
        """Extract and record token usage from an LLM response."""
        if not self.token_tracker:
            return
        from cliver.token_tracker import extract_usage

        usage = extract_usage(response)
        if usage.total_tokens > 0:
            model_name = model
            if not model_name:
                _model = self._get_llm_model(None)
                model_name = _model.name if _model else "unknown"
            self.token_tracker.record(model_name, usage)

    def get_mcp_caller(self) -> MCPServersCaller:
        return self.mcp_caller

    def _select_llm_engine(self, model: str = None) -> LLMInferenceEngine:
        _model = self._get_llm_model(model)
        if not _model:
            raise RuntimeError(f"No model named {model}.")
        if _model.name in self.llm_engines:
            llm_engine = self.llm_engines[_model.name]
        else:
            llm_engine = create_llm_engine(_model, user_agent=self.user_agent, agent_name=self.agent_name)
            self.llm_engines[_model.name] = llm_engine
        return llm_engine

    def _get_llm_model(self, model: str | None) -> ModelConfig:
        _model = None
        if model:
            _model = self.llm_models.get(model)
        elif self.default_model:
            _model = self.llm_models.get(self.default_model)
        return _model

    def get_llm_engine(self, model: str = None) -> LLMInferenceEngine:
        """
        Get the LLM engine for a specific model.

        Args:
            model: Model name to get engine for

        Returns:
            LLMInferenceEngine instance
        """
        return self._select_llm_engine(model)

    @staticmethod
    def _compute_required_capabilities(
        images: List[str] | None,
        audio_files: List[str] | None,
        video_files: List[str] | None,
        has_tools: bool,
    ) -> set:
        """Determine which capabilities this request requires."""
        from cliver.model_capabilities import ModelCapability

        caps = {ModelCapability.TEXT_TO_TEXT}
        if images:
            caps.add(ModelCapability.IMAGE_TO_TEXT)
        if audio_files:
            caps.add(ModelCapability.AUDIO_TO_TEXT)
        if video_files:
            caps.add(ModelCapability.VIDEO_TO_TEXT)
        if has_tools:
            caps.add(ModelCapability.TOOL_CALLING)
        return caps

    def _find_fallback_model(
        self,
        failed_model: str,
        required_capabilities: set,
        tried_models: set,
    ) -> str | None:
        """Find the next configured model with matching capabilities.

        Iterates self.llm_models in insertion order (config.yaml order).
        Skips models already tried, session-excluded models, and models
        missing required capabilities.
        """
        for name, config in self.llm_models.items():
            if name in tried_models or name in self.excluded_models:
                continue
            model_caps = config.get_capabilities()
            if required_capabilities.issubset(model_caps):
                return name
        return None

    async def _compress_for_retry(
        self,
        messages: List[BaseMessage],
        llm_engine: LLMInferenceEngine,
        model_config,
    ) -> List[BaseMessage]:
        """Compress conversation messages to fit within context window.

        Separates system messages from conversation, compresses the conversation,
        then reassembles. Used when context overflow is detected.
        """
        from cliver.conversation_compressor import ConversationCompressor, get_context_window

        context_window = get_context_window(model_config)
        compressor = ConversationCompressor(context_window)

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if not conv_msgs:
            return messages

        compressed_conv = await compressor.compress(conv_msgs, llm_engine)
        return system_msgs + compressed_conv

    async def _maybe_learn_skill(self, user_input: str, result: BaseMessage) -> None:
        """Post-task hook: trigger skill review if the task was complex.

        Runs asynchronously after the main task completes. Uses a low-iteration
        LLM call to evaluate whether to create a reusable SKILL.md.
        """
        try:
            from cliver.skill_reviewer import maybe_review_for_skill

            # Build a brief summary from the user input + result
            result_text = str(result.content)[:500] if result and result.content else ""
            task_summary = f"User asked: {user_input[:300]}\nResult: {result_text}"

            skills_dir = None
            if self.agent_profile:
                from cliver.util import get_config_dir

                skills_dir = get_config_dir() / "skills"

            skill_name = await maybe_review_for_skill(
                task_executor=self,
                tool_call_count=self._tool_call_count,
                task_summary=task_summary,
                skills_dir=skills_dir,
            )
            if skill_name:
                logger.info("Auto-learned skill: %s", skill_name)
        except Exception as e:
            # Never let skill review crash the main task flow
            logger.debug("Skill auto-learning failed (non-fatal): %s", e)

    def process_user_input_sync(
        self,
        user_input: str,
        images: List[str] = None,
        audio_files: List[str] = None,
        video_files: List[str] = None,
        files: List[str] = None,  # General files for tools like code interpreter
        max_iterations: int = 50,
        confirm_tool_exec: Optional[Callable[[str], bool]] = None,
        model: str = None,
        system_message_appender: Optional[Callable[[], str]] = None,
        filter_tools: Optional[Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]] = None,
        enhance_prompt: Optional[Callable[[str, MCPServersCaller], Awaitable[list[BaseMessage]]]] = None,
        tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]] = None,
        template: Optional[str] = None,
        params: dict = None,
        options: Dict[str, Any] = None,
        conversation_history: Optional[List[BaseMessage]] = None,
        timeout_s: Optional[int] = None,
        auto_fallback: bool = True,
        on_pending_input: Optional[Callable[[], Optional[str]]] = None,
    ) -> BaseMessage:
        return asyncio.run(
            self.process_user_input(
                user_input,
                images,
                audio_files,
                video_files,
                files,
                max_iterations,
                confirm_tool_exec,
                model,
                system_message_appender,
                filter_tools,
                enhance_prompt,
                tool_error_check,
                template,
                params,
                options,
                conversation_history,
                timeout_s=timeout_s,
                auto_fallback=auto_fallback,
                on_pending_input=on_pending_input,
            )
        )

    async def stream_user_input(
        self,
        user_input: str,
        images: List[str] = None,
        audio_files: List[str] = None,
        video_files: List[str] = None,
        files: List[str] = None,  # General files for tools like code interpreter
        max_iterations: int = 50,
        confirm_tool_exec: Optional[Callable[[str], bool]] = None,
        model: str = None,
        system_message_appender: Optional[Callable[[], str]] = None,
        filter_tools: Optional[Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]] = None,
        enhance_prompt: Optional[Callable[[str, MCPServersCaller], Awaitable[list[BaseMessage]]]] = None,
        tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]] = None,
        template: Optional[str] = None,
        params: dict = None,
        options: Dict[str, Any] = None,
        conversation_history: Optional[List[BaseMessage]] = None,
        timeout_s: Optional[int] = None,
        auto_fallback: bool = True,
        on_pending_input: Optional[Callable[[], Optional[str]]] = None,
    ) -> AsyncIterator[BaseMessageChunk]:
        """
        Stream user input through the LLM, handling tool calls if needed.
        Args:
            user_input (str): The user input.
            images (List[str]): List of image file paths to send with the message.
            audio_files (List[str]): List of audio file paths to send with the message.
            video_files (List[str]): List of video file paths to send with the message.
            files (List[str]): List of general file paths to upload for tools like code interpreter.
            max_iterations (int): The maximum number of iterations.
            confirm_tool_exec: Optional callback for tool execution confirmation. Receives a prompt
                              string, returns True to proceed or False to cancel. None = no confirmation.
            model (str): The model to use, the default one will be used if not specified.
            system_message_appender: The system message appender function.
            filter_tools: The function that filters tool calls.
            enhance_prompt: The function that enhances the prompt. This works alongside the default function
                           that reads Cliver.md for context.
            tool_error_check: The function that checks tool errors. The returned string will be the tool error message
                              sent back to LLM if the first returned value is True.
            template: Template name to apply.
            params: Parameters for templates.
            options: Dictionary of additional options to override LLM configurations.
        """

        # Route generation-only models directly (skip Re-Act loop)
        _model_config = self._get_llm_model(model)
        if _model_config:
            _caps = _model_config.get_capabilities()
            if ModelCapability.TEXT_TO_IMAGE in _caps and ModelCapability.TEXT_TO_TEXT not in _caps:
                result = await self.generate_image(user_input, model, **(options or {}))
                yield AIMessageChunk(content=result.content, additional_kwargs=result.additional_kwargs)
                return

        (
            llm_engine,
            llm_tools,
            messages,
        ) = await self._prepare_messages_and_tools(
            enhance_prompt,
            filter_tools,
            model,
            system_message_appender,
            user_input,
            images,
            audio_files,
            video_files,
            files,
            template,
            params,
            conversation_history,
        )

        # Reset per-call state
        self._generated_media = []

        required_caps = self._compute_required_capabilities(
            images,
            audio_files,
            video_files,
            has_tools=bool(llm_tools),
        )
        deadline = (time.monotonic() + timeout_s) if timeout_s else None

        async for chunk in self._stream_messages(
            llm_engine,
            model,
            messages,
            max_iterations,
            0,
            llm_tools,
            confirm_tool_exec,
            tool_error_check,
            options=options,
            deadline=deadline,
            required_capabilities=required_caps,
            auto_fallback=auto_fallback,
            tried_models={model} if model else set(),
            on_pending_input=on_pending_input,
        ):
            yield chunk

        # Yield a final chunk with any media generated by tools
        if self._generated_media:
            yield AIMessageChunk(
                content="",
                additional_kwargs={"media_content": self._generated_media},
            )

    async def _prepare_messages_and_tools(
        self,
        enhance_prompt,
        filter_tools,
        model,
        system_message_appender,
        user_input,
        images=None,
        audio_files=None,
        video_files=None,
        files=None,  # General files for tools like code interpreter
        template=None,
        params=None,
        conversation_history=None,
    ):
        # Check file upload capability early, before any processing
        if files:
            logger.debug(f"_prepare_messages_and_tools called with files: {files}")
            llm_engine = self._select_llm_engine(model)
            if hasattr(llm_engine, "config"):
                capabilities = llm_engine.config.get_capabilities()
                from cliver.model_capabilities import ModelCapability

                if ModelCapability.FILE_UPLOAD not in capabilities:
                    logger.info("File upload is not supported for this model. Will use content embedding as fallback.")

        llm_engine = self._select_llm_engine(model)
        logger.debug(f"Selected LLM engine: {type(llm_engine)}")
        # Build a single system prompt with all sections
        system_parts = [llm_engine.system_message()]
        if system_message_appender:
            system_message_extra = system_message_appender()
            if system_message_extra and len(system_message_extra) > 0:
                system_parts.append(system_message_extra)

        # Merge context from CLAUDE.md / Cliver.md etc.
        default_enhanced_messages = await default_enhance_prompt(user_input, self.mcp_caller)
        for msg in default_enhanced_messages or []:
            if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
                system_parts.append(msg.content)

        # Merge agent identity and memory
        if self.agent_profile:
            identity_content = self.agent_profile.load_identity()
            if identity_content:
                system_parts.append(f"# Identity Profile\n\n{identity_content}")

            memory_content = self.agent_profile.load_memory()
            if memory_content:
                system_parts.append(f"# Agent Memory\n\n{memory_content}")

        messages: list[BaseMessage] = [
            SystemMessage(content="\n\n".join(system_parts)),
        ]

        llm_tools: List[BaseTool] = []
        # mcp_tools are langchain BaseTool coming from MCP server, the name follows 'mcp_server_name#tool_name'
        mcp_tools: List[BaseTool] = await self.mcp_caller.get_mcp_tools()
        if filter_tools:
            mcp_tools = await filter_tools(user_input, mcp_tools)
        if mcp_tools:
            llm_tools.extend(mcp_tools)
        # Include builtin tools filtered by relevance to user input
        _builtin_tools = tool_registry.get_tools(user_input=user_input)
        if _builtin_tools:
            llm_tools.extend(_builtin_tools)

        # Apply template if provided
        if template:
            messages = apply_template(user_input, messages, template, params)

        # Apply user-provided enhancement function if provided
        if enhance_prompt:
            user_enhanced_messages = await enhance_prompt(user_input, self.mcp_caller)
            messages.extend(user_enhanced_messages)

        # Load media files if provided
        media_content = []

        # Load image files
        if images:
            logger.info(f"loading images: {images}")
            for image_path in images:
                try:
                    media_content.append(load_media_file(image_path))
                except Exception as e:
                    logger.warning(f"Could not load image file {image_path}: {e}")

        # Load audio files
        if audio_files:
            for audio_path in audio_files:
                try:
                    media_content.append(load_media_file(audio_path))
                except Exception as e:
                    logger.warning(f"Could not load audio file {audio_path}: {e}")

        # Load video files
        if video_files:
            for video_path in video_files:
                try:
                    media_content.append(load_media_file(video_path))
                except Exception as e:
                    logger.warning(f"Could not load video file {video_path}: {e}")

        # Handle file uploads for tools like code interpreter
        uploaded_file_ids = []
        embedded_files_content = []
        if files:
            logger.info(f"Processing file uploads for files: {files}")
            # Check if the model supports file uploads through its capabilities
            file_upload_supported = False
            model_config = self._get_llm_model(model)
            if model_config:
                capabilities = model_config.get_capabilities()
                from cliver.model_capabilities import ModelCapability

                file_upload_supported = ModelCapability.FILE_UPLOAD in capabilities

            if not file_upload_supported:
                logger.info(
                    "File upload is not supported for this model. Will embed file contents in the prompt instead."
                )
                # Fallback: embed file contents directly in the prompt
                for file_path in files:
                    try:
                        # Import here to avoid circular imports
                        from cliver.util import read_file_content

                        file_content = read_file_content(file_path)
                        embedded_files_content.append((file_path, file_content))
                        logger.info(f"Embedded content of file {file_path} in prompt")
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path} for embedding: {e}")
            else:
                # Original file upload logic
                for file_path in files:
                    try:
                        # Check if this is an OpenAI engine that supports file uploads
                        if hasattr(llm_engine, "upload_file"):
                            file_id = llm_engine.upload_file(file_path)
                            if file_id:
                                uploaded_file_ids.append(file_id)
                                logger.info(f"Uploaded file {file_path} with ID {file_id}")
                            else:
                                logger.warning(f"Failed to upload file {file_path}")
                        else:
                            logger.info(f"LLM engine doesn't support file uploads, skipping {file_path}")
                    except Exception as e:
                        logger.warning(
                            f"Could not upload file {file_path}: {e}. Please check the "
                            f"configuration if the capability is enabled."
                        )
                logger.info(f"Completed file uploads. Uploaded file IDs: {uploaded_file_ids}")

        # Insert conversation history (prior turns) before the current user input
        if conversation_history:
            messages.extend(conversation_history)

        # Add the user input with media content and file references
        if media_content or uploaded_file_ids or embedded_files_content:
            # Create a human message with media content and file references
            content_parts = [{"type": "text", "text": user_input}]

            # Add media content using shared utility function
            from cliver.media import add_media_content_to_message_parts

            add_media_content_to_message_parts(content_parts, media_content)

            # Add file references for uploaded files
            if uploaded_file_ids:
                content_parts.append(
                    {
                        "type": "text",
                        "text": f"\n\nUploaded files for reference: {', '.join(uploaded_file_ids)}",
                    }
                )

            # Add embedded file contents for models that don't support file uploads
            if embedded_files_content:
                content_parts.append(
                    {
                        "type": "text",
                        "text": "\n\nThe following files have been provided for context:",
                    }
                )
                for file_path, file_content in embedded_files_content:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"\n\nFile: {file_path}\nContent:\n```\n{file_content}\n```",
                        }
                    )

            human_message = HumanMessage(content=content_parts)
            messages.append(human_message)
        else:
            # Add the user input as a regular message
            messages.append(HumanMessage(content=user_input))

        return llm_engine, llm_tools, messages

    # This is the method that can be used out of box
    async def process_user_input(
        self,
        user_input: str,
        images: List[str] = None,
        audio_files: List[str] = None,
        video_files: List[str] = None,
        files: List[str] = None,  # General files for tools like code interpreter
        max_iterations: int = 50,
        confirm_tool_exec: Optional[Callable[[str], bool]] = None,
        model: str = None,
        system_message_appender: Optional[Callable[[], str]] = None,
        filter_tools: Optional[Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]] = None,
        enhance_prompt: Optional[Callable[[str, MCPServersCaller], Awaitable[list[BaseMessage]]]] = None,
        tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]] = None,
        template: Optional[str] = None,
        params: dict = None,
        options: Dict[str, Any] = None,
        conversation_history: Optional[List[BaseMessage]] = None,
        timeout_s: Optional[int] = None,
        auto_fallback: bool = True,
        on_pending_input: Optional[Callable[[], Optional[str]]] = None,
    ) -> BaseMessage:
        """
        Process user input through the LLM, handling tool calls if needed.
        Args:
            user_input (str): The user input.
            images (List[str]): List of image file paths to send with the message.
            audio_files (List[str]): List of audio file paths to send with the message.
            video_files (List[str]): List of video file paths to send with the message.
            files (List[str]): List of general file paths to upload for tools like code interpreter.
            max_iterations (int): The maximum number of iterations.
            confirm_tool_exec: Optional callback for tool execution confirmation. Receives a prompt
                              string, returns True to proceed or False to cancel. None = no confirmation.
            model (str): The model to use, the default one will be used if not specified.
            system_message_appender: The system message appender function.
            filter_tools: The function that filters tool calls.
            enhance_prompt: The function that enhances the prompt. This works alongside the default function
                           that reads Cliver.md for context.
            tool_error_check: The function that checks tool errors. The returned string will be the tool error message
                              sent back to LLM if the first returned value is True.
            template: Template name to apply.
            params: Parameters for templates.
            options: Additional options for LLM inference that can override what the ModelConfig is defined.
        """

        # Route generation-only models directly (skip Re-Act loop)
        _model_config = self._get_llm_model(model)
        if _model_config:
            _caps = _model_config.get_capabilities()
            if ModelCapability.TEXT_TO_IMAGE in _caps and ModelCapability.TEXT_TO_TEXT not in _caps:
                return await self.generate_image(user_input, model, **(options or {}))

        (
            llm_engine,
            llm_tools,
            messages,
        ) = await self._prepare_messages_and_tools(
            enhance_prompt,
            filter_tools,
            model,
            system_message_appender,
            user_input,
            images,
            audio_files,
            video_files,
            files,
            template,
            params,
            conversation_history,
        )

        # Reset per-call state
        self._tool_call_count = 0
        self._generated_media = []

        required_caps = self._compute_required_capabilities(
            images,
            audio_files,
            video_files,
            has_tools=bool(llm_tools),
        )
        deadline = (time.monotonic() + timeout_s) if timeout_s else None

        result = await self._process_messages(
            llm_engine,
            model,
            messages,
            max_iterations,
            0,
            llm_tools,
            confirm_tool_exec,
            tool_error_check,
            options=options,
            deadline=deadline,
            required_capabilities=required_caps,
            auto_fallback=auto_fallback,
            tried_models={model} if model else set(),
            on_pending_input=on_pending_input,
        )

        # Attach any media generated by tools (e.g., ImageGenerate) to the final response
        if self._generated_media and isinstance(result, AIMessage):
            kwargs = dict(result.additional_kwargs) if result.additional_kwargs else {}
            kwargs["media_content"] = self._generated_media
            result = AIMessage(content=result.content, additional_kwargs=kwargs)

        # Post-task: trigger skill auto-learning review if task was complex
        await self._maybe_learn_skill(user_input, result)

        return result

    async def _process_messages(
        self,
        llm_engine: LLMInferenceEngine,
        model: str,
        messages: List[BaseMessage],
        max_iterations: int,
        current_iteration: int,
        mcp_tools: list[BaseTool],
        confirm_tool_exec: Optional[Callable[[str], bool]],
        tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]] = None,
        options: Dict[str, Any] = None,
        deadline: Optional[float] = None,
        required_capabilities: Optional[set] = None,
        auto_fallback: bool = True,
        tried_models: Optional[set] = None,
        on_pending_input: Optional[Callable[[], Optional[str]]] = None,
    ) -> BaseMessage:
        """Handle processing messages with tool calling using a while loop."""

        iteration = current_iteration
        consecutive_errors = 0
        retries = 0
        max_retries = 3
        compressed_for: set = set()
        tried = tried_models.copy() if tried_models else set()
        current_model = model
        total_attempts = 0
        max_total_attempts = max_iterations * 3  # hard ceiling including retries/failovers

        try:
            while iteration < max_iterations:
                total_attempts += 1
                if total_attempts > max_total_attempts:
                    return AIMessage(content="Stopped: exceeded maximum total attempts.")
                # Check deadline before each iteration
                if deadline is not None and time.monotonic() >= deadline:
                    raise TaskTimeoutError(
                        "Task timed out",
                        partial_result=_extract_last_content(messages),
                    )

                options = options or {}

                # Re-inject current plan state so the LLM stays on track
                _inject_plan_context(messages)

                # Drain any pending user input
                _drain_pending_input(messages, on_pending_input)

                # Proactive context overflow check
                if current_model not in compressed_for:
                    try:
                        from cliver.conversation_compressor import estimate_tokens, get_context_window

                        model_config = self._get_llm_model(current_model)
                        if model_config:
                            context_window = get_context_window(model_config)
                            approx_tokens = estimate_tokens(messages)
                            if approx_tokens > context_window * 0.9:
                                logger.warning(
                                    "Context approaching limit (%d/%d tokens), compressing",
                                    approx_tokens,
                                    context_window,
                                )
                                compressed_for.add(current_model)
                                messages = await self._compress_for_retry(messages, llm_engine, model_config)
                    except Exception as comp_err:
                        logger.warning("Proactive compression failed: %s", comp_err)

                # Sanitize messages before sending to LLM
                _sanitize_messages(messages)

                # Pace calls according to provider rate limit
                await self._wait_for_rate_limit(current_model)

                try:
                    response = await llm_engine.infer(messages, mcp_tools, **options)
                except TaskTimeoutError:
                    raise
                except Exception as e:
                    from cliver.llm.error_classifier import ErrorAction, classify_error

                    classified = classify_error(e)
                    logger.info(
                        "LLM error: reason=%s action=%s model=%s",
                        classified.reason,
                        classified.action.value,
                        current_model,
                    )

                    # RETRY: transient error
                    if classified.action == ErrorAction.RETRY and retries < max_retries:
                        retries += 1
                        base_delay = 1.0
                        delay = min(base_delay * (2**retries), 30.0)
                        jitter = random.uniform(0, delay * 0.5)
                        self._emit_tool_event(
                            ToolEvent(
                                event_type=ToolEventType.MODEL_RETRY,
                                tool_name=current_model,
                                result=f"{classified.reason} (attempt {retries}/{max_retries})",
                            )
                        )
                        await asyncio.sleep(delay + jitter)
                        continue

                    # COMPRESS: context overflow
                    if classified.should_compress and current_model not in compressed_for:
                        compressed_for.add(current_model)
                        self._emit_tool_event(
                            ToolEvent(
                                event_type=ToolEventType.MODEL_COMPRESS,
                                tool_name=current_model,
                            )
                        )
                        try:
                            model_config = self._get_llm_model(current_model)
                            messages = await self._compress_for_retry(messages, llm_engine, model_config)
                            retries = 0
                            continue
                        except Exception as comp_err:
                            logger.warning("Compression failed: %s", comp_err)

                    # FAILOVER: switch model — on explicit FAILOVER action,
                    # OR when retries are exhausted for any error type
                    should_failover = classified.action == ErrorAction.FAILOVER or (
                        classified.action == ErrorAction.RETRY and retries >= max_retries
                    )
                    if should_failover and auto_fallback:
                        tried.add(current_model)
                        next_model = self._find_fallback_model(
                            current_model,
                            required_capabilities or set(),
                            tried,
                        )
                        if next_model:
                            reason = classified.reason
                            if retries >= max_retries:
                                reason = f"{reason} (retries exhausted)"
                            logger.info("Falling back from %s to %s (%s)", current_model, next_model, reason)
                            self._emit_tool_event(
                                ToolEvent(
                                    event_type=ToolEventType.MODEL_FALLBACK,
                                    tool_name=next_model,
                                    result=f"from {current_model} ({reason})",
                                )
                            )
                            llm_engine = self._select_llm_engine(next_model)
                            current_model = next_model
                            retries = 0
                            continue

                    # No recovery possible
                    friendly = get_friendly_error_message(e, "LLM inference")
                    return AIMessage(content=friendly)

                logger.debug(f"LLM response: {response}")

                # Guard against None response from LLM engine
                if response is None:
                    return AIMessage(content="Error: LLM returned empty response.")

                # Detect error responses from the engine (connection failures, auth errors, etc.)
                # These are returned as AIMessage with additional_kwargs={"type": "error"}
                if _is_error_response(response):
                    return response

                # Track token usage from this LLM call
                self._track_tokens(response, current_model)

                raw_tool_calls = llm_engine.parse_tool_calls(response, current_model)
                tool_calls = normalize_tool_calls(raw_tool_calls) if raw_tool_calls else None
                if tool_calls:
                    stop, result = await self._execute_tool_calls(
                        tool_calls,
                        messages,
                        confirm_tool_exec,
                        tool_error_check,
                        llm_response=response,
                    )
                    if stop:
                        return AIMessage(content=result)

                    # Track consecutive error iterations to detect error loops
                    if _has_tool_errors(tool_calls, messages):
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            return AIMessage(
                                content="Stopping: tool calls failed repeatedly. "
                                "Please check the tool arguments or try a different approach."
                            )
                    else:
                        consecutive_errors = 0

                    retries = 0
                    iteration += 1
                    continue
                # If no tool calls, return the response
                return response

            return AIMessage(content="Reached maximum number of iterations without a final answer.")
        finally:
            # Clean up browser session if active
            try:
                from cliver.tools.browser_action import close_browser_session

                await close_browser_session()
            except Exception:
                pass

    # returns a tuple to indicate if there is error occurs which indicates stop and a string message
    # this may execute multiple tools in sequence in one iteration of response
    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        messages: List[BaseMessage],
        confirm_tool_exec: Optional[Callable[[str], bool]],
        tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]],
        llm_response: Optional[BaseMessage] = None,
    ) -> Tuple[bool, str | None]:
        """Execute tool calls and return a Tuple with sent bool and result.

        Tool execution errors are sent back to the LLM as ToolMessage responses
        so it can self-correct (e.g., fix arguments, try a different tool).
        Only user cancellation or fatal exceptions stop the agent loop.

        Args:
            confirm_tool_exec: Optional callback that receives a prompt string
                and returns True to proceed or False to cancel.
                When None, tools run without confirmation.
        """
        try:
            # Track tool calls for skill auto-learning
            self._tool_call_count += len(tool_calls)

            for tool_call in tool_calls:
                mcp_server = tool_call.get("mcp_server", "")
                tool_name = tool_call.get("tool_name", "")
                args = tool_call.get("args") or {}
                tool_call_id = tool_call.get("tool_call_id", "")
                if not tool_name:
                    logger.warning("Skipping tool call with empty name: %s", tool_call)
                    continue
                full_tool_name = f"{mcp_server}#{tool_name}" if mcp_server else tool_name

                # Emit start event early so the CLI can stop the spinner
                # before any permission dialog or tool execution
                self._emit_tool_event(
                    ToolEvent(
                        event_type=ToolEventType.TOOL_START,
                        tool_name=full_tool_name,
                        tool_call_id=tool_call_id,
                        args=args,
                    )
                )

                # Permission / confirmation gate
                denied = await self._check_permission(
                    full_tool_name, args, tool_call_id, messages, llm_response, confirm_tool_exec
                )
                if denied is not None:
                    # Emit end event so spinner restarts for next LLM iteration
                    self._emit_tool_event(
                        ToolEvent(
                            event_type=ToolEventType.TOOL_END,
                            tool_name=full_tool_name,
                            tool_call_id=tool_call_id,
                            result="denied",
                        )
                    )
                    return denied

                # Append the tool execution to messages using AIMessage with tool_calls.
                # Preserve additional_kwargs from the original LLM response on the
                # first tool call so that provider-specific fields like DeepSeek's
                # reasoning_content are retained in conversation history.
                extra_kwargs: Dict[str, Any] = {}
                if llm_response is not None:
                    resp_kwargs = getattr(llm_response, "additional_kwargs", None)
                    if resp_kwargs:
                        extra_kwargs = dict(resp_kwargs)
                    # Only attach to the first tool call message
                    llm_response = None
                tool_execution_message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": full_tool_name,
                            "args": args,
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                    additional_kwargs=extra_kwargs,
                )
                messages.append(tool_execution_message)

                # Execute the tool and capture the result
                start_time = time.monotonic()
                tool_result = await self._execute_single_tool(mcp_server, tool_name, args, full_tool_name)
                duration_ms = (time.monotonic() - start_time) * 1000

                # Format and append result as ToolMessage
                tool_result_content = _format_tool_result(tool_result)
                messages.append(ToolMessage(content=tool_result_content, tool_call_id=tool_call_id))

                # Check for errors and emit appropriate event
                error_checker = tool_error_check or _tool_error_check_internal
                has_error, error_msg = error_checker(tool_name, tool_result)
                if has_error:
                    logger.warning(f"Tool '{full_tool_name}' returned error: {error_msg}")
                    self._emit_tool_event(
                        ToolEvent(
                            event_type=ToolEventType.TOOL_ERROR,
                            tool_name=full_tool_name,
                            tool_call_id=tool_call_id,
                            error=error_msg,
                            duration_ms=duration_ms,
                        )
                    )
                else:
                    self._emit_tool_event(
                        ToolEvent(
                            event_type=ToolEventType.TOOL_END,
                            tool_name=full_tool_name,
                            tool_call_id=tool_call_id,
                            result=tool_result_content,
                            duration_ms=duration_ms,
                        )
                    )

            # All tool calls processed, continue to next LLM iteration
            return False, "Tool calls executed successfully"
        except Exception as e:
            logger.error(f"Error processing tool call: {str(e)}", exc_info=True)
            # Don't hard-stop on tool execution exceptions — feed the error back
            # to the LLM so it can self-correct, and let consecutive_errors track it.
            friendly_error_msg = get_friendly_error_message(e, "Tool call processing")
            messages.append(
                ToolMessage(
                    content=f"Error: {friendly_error_msg}",
                    tool_call_id=tool_calls[-1].get("tool_call_id", "") if tool_calls else "",
                )
            )
            return False, friendly_error_msg

    async def _check_permission(
        self,
        full_tool_name: str,
        args: dict,
        tool_call_id: str,
        messages: List[BaseMessage],
        llm_response: Optional[BaseMessage],
        confirm_tool_exec: Optional[Callable[[str], bool]],
    ) -> Optional[Tuple[bool, str | None]]:
        """Check permission for a tool call.

        Returns None if allowed (proceed), or a (stop, result) tuple
        to return from _execute_tool_calls.

        Resolution: PermissionManager > confirm_tool_exec > auto-allow.
        """
        if self.permission_manager is not None:
            decision = self.permission_manager.check(full_tool_name, args)
            if decision == PermissionDecision.DENY:
                self._append_denied_tool_message(full_tool_name, args, tool_call_id, messages, llm_response)
                return False, None
            elif decision == PermissionDecision.ASK:
                user_choice = self._prompt_user_permission(full_tool_name, args)
                if user_choice in ("deny", "deny_always"):
                    if user_choice == "deny_always":
                        self.permission_manager.grant_session(full_tool_name, PermissionAction.DENY)
                    self._append_denied_tool_message(full_tool_name, args, tool_call_id, messages, llm_response)
                    return False, None
                elif user_choice == "allow_always":
                    self.permission_manager.grant_session(full_tool_name, PermissionAction.ALLOW)
            return None

        if confirm_tool_exec is not None:
            if not confirm_tool_exec(f"Execute tool: {full_tool_name}? (y/n): "):
                return True, f"Stopped at tool execution: {full_tool_name}"

        return None

    def _prompt_user_permission(self, full_tool_name: str, args: dict) -> str:
        """Prompt user for permission. Returns allow/allow_always/deny/deny_always."""
        if self.on_permission_prompt is not None:
            return self.on_permission_prompt(full_tool_name, args)
        return _default_permission_prompt(full_tool_name, args)

    def _append_denied_tool_message(
        self,
        full_tool_name: str,
        args: dict,
        tool_call_id: str,
        messages: List[BaseMessage],
        llm_response: Optional[BaseMessage],
    ):
        """Append AIMessage + ToolMessage for a denied tool call."""
        extra_kwargs: Dict[str, Any] = {}
        if llm_response is not None:
            resp_kwargs = getattr(llm_response, "additional_kwargs", None)
            if resp_kwargs:
                extra_kwargs = dict(resp_kwargs)
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": full_tool_name,
                        "args": args,
                        "id": tool_call_id,
                        "type": "tool_call",
                    }
                ],
                additional_kwargs=extra_kwargs,
            )
        )
        messages.append(
            ToolMessage(
                content=f"Permission denied: The user denied execution of '{full_tool_name}'. "
                f"Do NOT retry this tool. Inform the user that the action was skipped "
                f"and ask how they would like to proceed.",
                tool_call_id=tool_call_id,
            )
        )

    async def _execute_single_tool(
        self,
        mcp_server: str,
        tool_name: str,
        args: Dict[str, Any],
        full_tool_name: str,
    ) -> List[Dict[str, Any]]:
        """Execute a single tool call, returning a result list.

        Handles tool-not-found with suggestions and wraps execution
        exceptions as error results so the LLM can recover.
        """
        try:
            if not mcp_server or mcp_server == "" or mcp_server == "builtin":
                # Check if tool exists before executing
                tool = tool_registry.get_tool_by_name(tool_name)
                if tool is None:
                    suggestion = _suggest_similar_tool(tool_name, tool_registry.tool_names)
                    return [{"error": f"Tool '{tool_name}' not found.{suggestion}"}]
                return tool_registry.execute_tool(tool_name, args)
            else:
                return await retry_with_confirmation_async(
                    self.mcp_caller.call_mcp_server_tool,
                    mcp_server,
                    tool_name,
                    args,
                    confirm_on_retry=False,
                )
        except Exception as e:
            logger.error(f"Exception executing tool '{full_tool_name}': {e}", exc_info=True)
            return [{"error": f"Tool execution failed: {e}"}]

    async def _stream_messages(
        self,
        llm_engine: LLMInferenceEngine,
        model: str,
        messages: List[BaseMessage],
        max_iterations: int,
        current_iteration: int,
        mcp_tools: list[BaseTool],
        confirm_tool_exec: Optional[Callable[[str], bool]],
        tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]],
        options: Dict[str, Any] = None,
        deadline: Optional[float] = None,
        required_capabilities: Optional[set] = None,
        auto_fallback: bool = True,
        tried_models: Optional[set] = None,
        on_pending_input: Optional[Callable[[], Optional[str]]] = None,
    ) -> AsyncIterator[BaseMessageChunk]:
        """Handle streaming messages with tool calling."""
        iteration = current_iteration
        consecutive_errors = 0
        retries = 0
        max_retries = 3
        compressed_for: set = set()
        tried = tried_models.copy() if tried_models else set()
        current_model = model
        total_attempts = 0
        max_total_attempts = max_iterations * 3  # hard ceiling including retries/failovers

        # keeps streaming unless got the final answer
        try:
            while iteration < max_iterations:
                total_attempts += 1
                if total_attempts > max_total_attempts:
                    # noinspection PyArgumentList
                    yield AIMessageChunk(content="Stopped: exceeded maximum total attempts.")
                    return
                # Check deadline before each iteration
                if deadline is not None and time.monotonic() >= deadline:
                    raise TaskTimeoutError(
                        "Task timed out",
                        partial_result=_extract_last_content(messages),
                    )
                # Re-inject current plan state so the LLM stays on track
                _inject_plan_context(messages)

                # Drain any pending user input
                _drain_pending_input(messages, on_pending_input)

                # Proactive context overflow check
                if current_model not in compressed_for:
                    try:
                        from cliver.conversation_compressor import estimate_tokens, get_context_window

                        model_config = self._get_llm_model(current_model)
                        if model_config:
                            context_window = get_context_window(model_config)
                            approx_tokens = estimate_tokens(messages)
                            if approx_tokens > context_window * 0.9:
                                logger.warning(
                                    "Context approaching limit (%d/%d tokens), compressing",
                                    approx_tokens,
                                    context_window,
                                )
                                compressed_for.add(current_model)
                                messages = await self._compress_for_retry(messages, llm_engine, model_config)
                    except Exception as comp_err:
                        logger.warning("Proactive compression failed: %s", comp_err)

                # Sanitize messages before sending to LLM
                _sanitize_messages(messages)

                # Pace calls according to provider rate limit
                await self._wait_for_rate_limit(current_model)

                # For proper tool call handling in streaming, we'll process differently
                tool_calls = None
                accumulated_chunks = None
                content_yielded = False
                was_thinking = False
                post_think_strip = False  # strip leading whitespace after </think>
                try:
                    options = options or {}
                    async for chunk in llm_engine.stream(messages, mcp_tools, **options):
                        # Accumulate chunks for tool call detection after stream ends
                        try:
                            if not accumulated_chunks:
                                accumulated_chunks = chunk
                            else:
                                accumulated_chunks = accumulated_chunks + chunk
                        except Exception as e:
                            logger.debug(f"Chunk accumulation error (non-fatal): {e}")
                            # If chunk addition fails, keep what we have and skip

                        # Filter out thinking/reasoning content from stream output.
                        # Reasoning may arrive in three forms:
                        # 1. additional_kwargs with reasoning_content (DeepSeek API)
                        # 2. Content block list with type='thinking' (Anthropic API)
                        # 3. <think>/<thinking> tags in string content (OpenAI-compat)
                        if llm_engine.supports_capability(ModelCapability.THINK_MODE):
                            # Check for structured reasoning in additional_kwargs
                            chunk_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
                            if chunk_kwargs.get("reasoning_content") or chunk_kwargs.get("reasoning"):
                                was_thinking = True
                                continue

                            # Check for Anthropic-style content blocks (list of dicts).
                            # Thinking blocks are filtered; text blocks are
                            # extracted as plain strings for downstream display.
                            chunk_content = getattr(chunk, "content", None)
                            if isinstance(chunk_content, list):
                                text_parts = []
                                has_thinking = False
                                for block in chunk_content:
                                    if isinstance(block, dict):
                                        if block.get("type") == "thinking":
                                            has_thinking = True
                                        elif block.get("type") == "text":
                                            text_parts.append(block.get("text", ""))
                                if has_thinking and not text_parts:
                                    was_thinking = True
                                    continue
                                # Convert list-of-blocks to plain string
                                joined = "".join(text_parts)
                                if has_thinking:
                                    was_thinking = False
                                # noinspection PyArgumentList
                                chunk = type(chunk)(content=joined)
                                chunk_content = joined

                            # Check for <think> tags in accumulated string content
                            if isinstance(chunk_content, str) or chunk_content is None:
                                acc_content = getattr(accumulated_chunks, "content", "") or ""
                                if isinstance(acc_content, str):
                                    chunks_content = acc_content
                                    if (
                                        len(chunks_content) <= 7 and "<think".startswith(chunks_content)
                                    ) or is_thinking(chunks_content):
                                        was_thinking = True
                                        continue
                                    # Exiting think mode: strip the think block from
                                    # accumulated text, yield only the clean remainder.
                                    if was_thinking:
                                        was_thinking = False
                                        from cliver.llm.llm_utils import remove_thinking_sections

                                        cleaned = remove_thinking_sections(chunks_content).lstrip("\n")
                                        if cleaned:
                                            # noinspection PyArgumentList
                                            chunk = type(chunk)(content=cleaned)
                                        else:
                                            post_think_strip = True
                                            continue

                            # Exiting think mode from Anthropic blocks
                            if was_thinking and isinstance(chunk_content, str):
                                was_thinking = False
                                post_think_strip = True

                            # Strip leading blank lines from subsequent chunks
                            # until we reach real content after a think block.
                            if post_think_strip and hasattr(chunk, "content") and isinstance(chunk.content, str):
                                stripped = chunk.content.lstrip("\n")
                                if not stripped:
                                    continue
                                post_think_strip = False
                                # noinspection PyArgumentList
                                chunk = type(chunk)(content=stripped)
                        if hasattr(chunk, "content") and chunk.content:
                            content_yielded = True
                        yield chunk

                    # Detect error responses from the engine
                    if accumulated_chunks and _is_error_response(accumulated_chunks):
                        return

                    # Track token usage from the accumulated streaming response
                    if accumulated_chunks:
                        self._track_tokens(accumulated_chunks, current_model)

                    if accumulated_chunks:
                        raw_tool_calls = llm_engine.parse_tool_calls(accumulated_chunks, current_model)
                        tool_calls = normalize_tool_calls(raw_tool_calls) if raw_tool_calls else None
                    else:
                        raw_tool_calls = None
                        tool_calls = None
                except TaskTimeoutError:
                    raise
                except Exception as e:
                    from cliver.llm.error_classifier import ErrorAction, classify_error

                    classified = classify_error(e)
                    logger.info(
                        "LLM streaming error: reason=%s action=%s model=%s",
                        classified.reason,
                        classified.action.value,
                        current_model,
                    )

                    # If content was already yielded to the user, don't retry
                    # (partial delivery — retrying would duplicate output)
                    if content_yielded:
                        logger.warning("Error after partial delivery, not retrying: %s", e)
                        return

                    # RETRY: transient error
                    if classified.action == ErrorAction.RETRY and retries < max_retries:
                        retries += 1
                        base_delay = 1.0
                        delay = min(base_delay * (2**retries), 30.0)
                        jitter = random.uniform(0, delay * 0.5)
                        self._emit_tool_event(
                            ToolEvent(
                                event_type=ToolEventType.MODEL_RETRY,
                                tool_name=current_model,
                                result=f"{classified.reason} (attempt {retries}/{max_retries})",
                            )
                        )
                        await asyncio.sleep(delay + jitter)
                        continue

                    # COMPRESS: context overflow
                    if classified.should_compress and current_model not in compressed_for:
                        compressed_for.add(current_model)
                        self._emit_tool_event(
                            ToolEvent(
                                event_type=ToolEventType.MODEL_COMPRESS,
                                tool_name=current_model,
                            )
                        )
                        try:
                            model_config = self._get_llm_model(current_model)
                            messages = await self._compress_for_retry(messages, llm_engine, model_config)
                            retries = 0
                            continue
                        except Exception as comp_err:
                            logger.warning("Compression failed: %s", comp_err)

                    # FAILOVER: switch model — on explicit FAILOVER action,
                    # OR when retries are exhausted for any error type
                    should_failover = classified.action == ErrorAction.FAILOVER or (
                        classified.action == ErrorAction.RETRY and retries >= max_retries
                    )
                    if should_failover and auto_fallback:
                        tried.add(current_model)
                        next_model = self._find_fallback_model(
                            current_model,
                            required_capabilities or set(),
                            tried,
                        )
                        if next_model:
                            reason = classified.reason
                            if retries >= max_retries:
                                reason = f"{reason} (retries exhausted)"
                            logger.info("Falling back from %s to %s (%s)", current_model, next_model, reason)
                            self._emit_tool_event(
                                ToolEvent(
                                    event_type=ToolEventType.MODEL_FALLBACK,
                                    tool_name=next_model,
                                    result=f"from {current_model} ({reason})",
                                )
                            )
                            llm_engine = self._select_llm_engine(next_model)
                            current_model = next_model
                            retries = 0
                            continue

                    # No recovery possible
                    friendly_error_msg = get_friendly_error_message(e, "LLM streaming")
                    # noinspection PyArgumentList
                    yield AIMessageChunk(
                        content=f"Error in streaming: {friendly_error_msg}",
                    )
                    return

                # If we found tool calls, execute them and continue after emitting the chunks
                if tool_calls:
                    stop, result = await self._execute_tool_calls(
                        tool_calls, messages, confirm_tool_exec, tool_error_check, llm_response=accumulated_chunks
                    )
                    if stop:
                        if result:
                            # noinspection PyArgumentList
                            yield AIMessageChunk(content=result)
                        return

                    # Track consecutive error iterations to detect error loops
                    if _has_tool_errors(tool_calls, messages):
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            # noinspection PyArgumentList
                            yield AIMessageChunk(
                                content="Stopping: tool calls failed repeatedly. "
                                "Please check the tool arguments or try a different approach."
                            )
                            return
                    else:
                        consecutive_errors = 0

                    retries = 0
                    iteration += 1
                    continue
                else:
                    # No tool calls found, we're done with this iteration
                    # If we reached here, the response is complete
                    return

            # If we've reached max iterations
            # noinspection PyArgumentList
            yield AIMessageChunk(content="Reached maximum number of iterations without a final answer.")
        finally:
            # Clean up browser session if active
            try:
                from cliver.tools.browser_action import close_browser_session

                await close_browser_session()
            except Exception:
                pass


def _is_error_response(response) -> bool:
    """Check if an LLM response represents an engine-level error.

    The base engine returns AIMessage with additional_kwargs={"type": "error"}
    on connection failures, auth errors, etc. These should NOT be fed back
    into the Re-Act loop as they'll just repeat the same failing call.
    """
    if response is None:
        return False
    kwargs = getattr(response, "additional_kwargs", None)
    if kwargs and kwargs.get("type") == "error":
        return True
    return False


def _has_tool_errors(tool_calls: List[Dict], messages: List[BaseMessage]) -> bool:
    """Check if the most recent tool results in messages contain errors."""
    # Look at the last N ToolMessages (matching the tool_calls count)
    tool_msg_count = len(tool_calls)
    recent_tool_msgs = [m for m in messages[-tool_msg_count * 2 :] if isinstance(m, ToolMessage)]
    if not recent_tool_msgs:
        return False
    return any(
        isinstance(m.content, str) and m.content.startswith("Error:") for m in recent_tool_msgs[-tool_msg_count:]
    )


# Sentinel content prefix used to identify plan-context messages so we can
# remove stale ones before injecting fresh state.
_PLAN_CONTEXT_PREFIX = "[Plan Status]"


def _inject_plan_context(messages: List[BaseMessage]) -> None:
    """Inject current todo plan state into the first SystemMessage.

    If a plan exists (via todo_write), appends the plan status to the
    existing system message so the LLM always knows where it is in its plan,
    even if earlier todo_write results have scrolled out of attention.

    Also adds a completion hint when all tasks are done.
    """
    from cliver.tools.todo_write import format_todo_summary, get_current_todos

    todos = get_current_todos()
    if not todos:
        return

    summary = format_todo_summary(todos)
    all_completed = all(item.get("status") == "completed" for item in todos)
    if all_completed:
        summary += "\n\nAll planned tasks are completed. Provide your final summary to the user."

    plan_block = f"\n\n{_PLAN_CONTEXT_PREFIX}\n{summary}"

    # Find the first SystemMessage and update it in-place
    for msg in messages:
        if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
            # Strip any previous plan block, then append fresh one
            content = msg.content
            plan_idx = content.find(f"\n\n{_PLAN_CONTEXT_PREFIX}")
            if plan_idx != -1:
                content = content[:plan_idx]
            msg.content = content + plan_block
            return

    # Fallback: no SystemMessage found (shouldn't happen), append one
    messages.append(SystemMessage(content=f"{_PLAN_CONTEXT_PREFIX}\n{summary}"))


def _drain_pending_input(
    messages: List[BaseMessage],
    on_pending_input: Optional[Callable[[], Optional[str]]],
) -> None:
    """Drain pending user input from callback and append as HumanMessages.

    This allows injecting user input between Re-Act iterations without
    breaking the conversation flow. The callback is called repeatedly until
    it returns None, with each result appended as a HumanMessage.

    Args:
        messages: The conversation message list to append to
        on_pending_input: Optional callback that returns pending input or None
    """
    if on_pending_input is None:
        return
    while True:
        pending = on_pending_input()
        if pending is None:
            break
        messages.append(HumanMessage(content=pending))
        logger.info("Injected pending user input (%d chars)", len(pending))


def _format_tool_result(tool_result) -> str:
    """Format tool result into a string for ToolMessage content.

    Handles various result formats:
    - list of dicts (standard): extract text/error/tool_result from first item
    - plain string: return as-is
    - None/empty: "(no output)"
    """
    if tool_result is None:
        return "(no output)"
    if isinstance(tool_result, str):
        return tool_result if tool_result else "(no output)"
    if isinstance(tool_result, list):
        if len(tool_result) == 0:
            return "(no output)"
        first_result = tool_result[0]
        if isinstance(first_result, dict):
            if "error" in first_result:
                return f"Error: {first_result['error']}"
            if "text" in first_result:
                return str(first_result["text"])
            if "tool_result" in first_result:
                return str(first_result["tool_result"])
        return str(tool_result)
    return str(tool_result)


def _tool_error_check_internal(tool_name: str, mcp_tool_result: list[Dict[str, Any]]) -> Tuple[bool, str | None]:
    """Check if a tool result contains errors. Returns (has_error, error_message)."""
    if not isinstance(mcp_tool_result, list):
        return False, None
    for result in mcp_tool_result:
        if isinstance(result, dict) and "error" in result:
            return (
                True,
                f"Tool '{tool_name}' error: {result['error']}",
            )
    return False, None


def _suggest_similar_tool(tool_name: str, available_names: list, max_suggestions: int = 3) -> str:
    """Suggest similar tool names using simple edit distance."""
    if not available_names:
        return ""

    # Score by common subsequence length (simple but effective)
    scored = []
    target = tool_name.lower()
    for name in available_names:
        candidate = name.lower()
        # Count matching characters in order
        common = sum(1 for c in target if c in candidate)
        # Penalise length difference
        score = common - abs(len(target) - len(candidate))
        scored.append((score, name))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [name for score, name in scored[:max_suggestions] if score > 0]

    if top:
        suggestions = ", ".join(f"'{n}'" for n in top)
        return f" Did you mean: {suggestions}?"
    return ""


def default_confirm_tool_execution(prompt: str) -> bool:
    """Default CLI-based tool execution confirmation via stdin.

    This is the default implementation used when no custom callback
    is provided. API consumers should provide their own callback
    to avoid stdin dependency.
    """
    while True:
        try:
            response = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False


def _default_permission_prompt(tool_name: str, args: dict) -> str:
    """Default stdin-based permission prompt.

    Returns one of: "allow", "allow_always", "deny", "deny_always"
    """
    # Extract a short resource description from args
    from cliver.permissions import get_tool_meta

    bare = tool_name.split("#")[-1] if "#" in tool_name else tool_name
    meta = get_tool_meta(bare)
    resource = str(args.get(meta.resource_param, "")) if meta.resource_param else ""

    print(f"  Permission required: {tool_name}")
    if resource:
        print(f"    Resource: {resource}")
    print("    [y]es / [n]o / [a]lways allow / [d]eny always")

    while True:
        try:
            response = input("    > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        if response in ("y", "yes"):
            return "allow"
        elif response in ("a", "always"):
            return "allow_always"
        elif response in ("n", "no"):
            return "deny"
        elif response in ("d", "deny"):
            return "deny_always"
