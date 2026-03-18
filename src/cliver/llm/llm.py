import asyncio
import logging
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
from cliver.llm.errors import get_friendly_error_message
from cliver.llm.llm_utils import is_thinking
from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
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
    if model.provider == "ollama":
        return OllamaLlamaInferenceEngine(model, user_agent=user_agent, agent_name=agent_name)
    elif model.provider == "openai":
        # Route to model-specific engines for providers with API quirks
        name = (model.name_in_provider or model.name).lower()
        if name.startswith("deepseek"):
            return DeepSeekInferenceEngine(model, user_agent=user_agent, agent_name=agent_name)
        return OpenAICompatibleInferenceEngine(model, user_agent=user_agent, agent_name=agent_name)
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


# Tool registry with keyword-based filtering, loaded on first use
tool_registry = ToolRegistry()


class TaskExecutor:
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
    ):
        self.llm_models = llm_models
        self.default_model = default_model
        self.user_agent = user_agent
        self.agent_name = agent_name
        self.on_tool_event = on_tool_event
        self.agent_profile = agent_profile  # AgentProfile for memory/identity
        self.token_tracker = token_tracker  # TokenTracker for usage auditing
        self.permission_manager = permission_manager
        self.on_permission_prompt = on_permission_prompt
        self.mcp_caller = MCPServersCaller(mcp_servers=mcp_servers)
        self.llm_engines: Dict[str, LLMInferenceEngine] = {}

        # Set the active profile so builtin tools (memory, etc.) can access it
        if agent_profile:
            from cliver.agent_profile import set_current_profile

            set_current_profile(agent_profile)

    def _emit_tool_event(self, event: ToolEvent) -> None:
        """Emit a tool event to the registered handler, if any."""
        if self.on_tool_event:
            try:
                self.on_tool_event(event)
            except Exception as e:
                logger.debug(f"Tool event handler error: {e}")

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

        # Since we've enhanced the infer and stream methods to handle multimedia,
        # we can always use the regular stream method
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
        ):
            yield chunk

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
        # Add system message to instruct the LLM about tool usage
        system_message = llm_engine.system_message()
        if system_message_appender:
            system_message_extra = system_message_appender()
            if system_message_extra and len(system_message_extra) > 0:
                system_message = f"{system_message}\n{system_message_extra}"

        # Create initial messages with system message
        messages: list[BaseMessage] = [
            SystemMessage(content=system_message),
        ]
        # Always apply the default enhancement function to get context from Cliver.md
        default_enhanced_messages = await default_enhance_prompt(user_input, self.mcp_caller)
        if default_enhanced_messages and len(default_enhanced_messages) > 0:
            messages.extend(default_enhanced_messages)

        # Inject agent identity and memory into the context
        if self.agent_profile:
            identity_content = self.agent_profile.load_identity()
            if identity_content:
                messages.append(SystemMessage(content=f"# Identity Profile\n\n{identity_content}"))

            memory_content = self.agent_profile.load_memory()
            if memory_content:
                messages.append(SystemMessage(content=f"# Agent Memory\n\n{memory_content}"))

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

        # Since we've enhanced the infer and stream methods to handle multimedia,
        # we can always use the regular infer method
        return await self._process_messages(
            llm_engine,
            model,
            messages,
            max_iterations,
            0,
            llm_tools,
            confirm_tool_exec,
            tool_error_check,
            options=options,
        )

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
    ) -> BaseMessage:
        """Handle processing messages with tool calling using a while loop."""

        iteration = current_iteration
        consecutive_errors = 0
        while iteration < max_iterations:
            options = options or {}

            # Re-inject current plan state so the LLM stays on track
            _inject_plan_context(messages)

            response = await llm_engine.infer(messages, mcp_tools, **options)
            logger.debug(f"LLM response: {response}")

            # Track token usage from this LLM call
            self._track_tokens(response, model)

            tool_calls = llm_engine.parse_tool_calls(response, model)
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

                iteration += 1
                continue
            # If no tool calls, return the response
            return response

        return AIMessage(content="Reached maximum number of iterations without a final answer.")

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
            for tool_call in tool_calls:
                mcp_server = tool_call["mcp_server"]
                tool_name = tool_call["tool_name"]
                args = tool_call["args"]
                tool_call_id = tool_call["tool_call_id"]
                full_tool_name = f"{mcp_server}#{tool_name}" if mcp_server else f"{tool_name}"

                # Permission / confirmation gate
                denied = await self._check_permission(
                    full_tool_name, args, tool_call_id, messages, llm_response, confirm_tool_exec
                )
                if denied is not None:
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

                # Emit start event
                self._emit_tool_event(
                    ToolEvent(
                        event_type=ToolEventType.TOOL_START,
                        tool_name=full_tool_name,
                        tool_call_id=tool_call_id,
                        args=args,
                    )
                )

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
                            result=tool_result_content[:200],  # truncate for display
                            duration_ms=duration_ms,
                        )
                    )

            # All tool calls processed, continue to next LLM iteration
            return False, "Tool calls executed successfully"
        except Exception as e:
            logger.error(f"Error processing tool call: {str(e)}", exc_info=True)
            friendly_error_msg = get_friendly_error_message(e, "Tool call processing")
            return True, friendly_error_msg

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
                content=f"Permission denied: {full_tool_name} is not allowed.",
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
    ) -> AsyncIterator[BaseMessageChunk]:
        """Handle streaming messages with tool calling."""
        iteration = current_iteration
        consecutive_errors = 0

        # keeps streaming unless got the final answer
        while iteration < max_iterations:
            # Re-inject current plan state so the LLM stays on track
            _inject_plan_context(messages)

            # For proper tool call handling in streaming, we'll process differently
            tool_calls = None
            accumulated_chunks = None
            try:
                # Create an internal streaming method that can detect tool calls early
                options = options or {}
                async for chunk in llm_engine.stream(messages, mcp_tools, **options):
                    # the chunk maybe part of the thinking content: '<thinking>xxx</thinking>'
                    if not accumulated_chunks:
                        accumulated_chunks = chunk
                    else:
                        accumulated_chunks = accumulated_chunks + chunk

                    # Filter out thinking/reasoning content from stream output.
                    # Reasoning may arrive via additional_kwargs (structured API field)
                    # or as <think>/<thinking> tags in content (raw model output).
                    if llm_engine.supports_capability(ModelCapability.THINK_MODE):
                        # Check for structured reasoning in additional_kwargs
                        chunk_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
                        if chunk_kwargs.get("reasoning_content") or chunk_kwargs.get("reasoning"):
                            continue
                        # Check for <think> tags in accumulated content
                        chunks_content = str(accumulated_chunks)
                        if (len(chunks_content) <= 7 and "<think".startswith(chunks_content)) or is_thinking(
                            chunks_content
                        ):
                            continue
                    yield chunk
                # Track token usage from the accumulated streaming response
                if accumulated_chunks:
                    self._track_tokens(accumulated_chunks, model)

                tool_calls = llm_engine.parse_tool_calls(accumulated_chunks, model)
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                friendly_error_msg = get_friendly_error_message(e, "LLM streaming")
                # Yield error message as a BaseMessageChunk (using AIMessageChunk)
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

                iteration += 1
                continue
            else:
                # No tool calls found, we're done with this iteration
                # If we reached here, the response is complete
                return

        # If we've reached max iterations
        # noinspection PyArgumentList
        yield AIMessageChunk(content="Reached maximum number of iterations without a final answer.")


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
    """Inject current todo plan state into the message list.

    If a plan exists (via todo_write), appends a SystemMessage with the
    current status so the LLM always knows where it is in its plan,
    even if earlier todo_write results have scrolled out of attention.

    Also adds a completion hint when all tasks are done.
    """
    from cliver.tools.todo_write import format_todo_summary, get_current_todos

    todos = get_current_todos()
    if not todos:
        return

    # Remove any previous plan-context message to avoid stacking
    messages[:] = [m for m in messages if not _is_plan_context_message(m)]

    summary = format_todo_summary(todos)

    # Check if all tasks are completed
    all_completed = all(item.get("status") == "completed" for item in todos)
    if all_completed:
        summary += "\n\nAll planned tasks are completed. Provide your final summary to the user."

    messages.append(SystemMessage(content=f"{_PLAN_CONTEXT_PREFIX}\n{summary}"))


def _is_plan_context_message(msg: BaseMessage) -> bool:
    """Check if a message is a plan-context injection."""
    return (
        isinstance(msg, SystemMessage) and isinstance(msg.content, str) and msg.content.startswith(_PLAN_CONTEXT_PREFIX)
    )


def _format_tool_result(tool_result: list) -> str:
    """Format tool result list into a string for ToolMessage content."""
    if isinstance(tool_result, list) and len(tool_result) > 0:
        first_result = tool_result[0]
        if isinstance(first_result, dict):
            # Error results: include the error message clearly
            if "error" in first_result:
                return f"Error: {first_result['error']}"
            if "text" in first_result:
                return first_result["text"]
            if "tool_result" in first_result:
                return first_result["tool_result"]
        return str(tool_result)
    return str(tool_result) if tool_result else "(no output)"


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
