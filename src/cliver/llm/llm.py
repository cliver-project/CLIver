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

# Create a logger for this module
logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
from cliver.mcp_server_caller import MCPServersCaller
from cliver.media import load_media_file
from cliver.prompt_enhancer import apply_skill_sets_and_template
from cliver.util import read_context_files, retry_with_confirmation_async


def create_llm_engine(model: ModelConfig) -> Optional[LLMInferenceEngine]:
    if model.provider == "ollama":
        return OllamaLlamaInferenceEngine(model)
    elif model.provider == "openai":
        return OpenAICompatibleInferenceEngine(model)
    return None


## TODO: we need to improve this to take consideration of user_input and even call some mcp tools
## TODO: We may need to parsed the structured context file and do embedding to get only related sections.
async def default_enhance_prompt(
    user_input: str, mcp_caller: MCPServersCaller
) -> list[BaseMessage]:
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


class TaskExecutor:
    """
    This is the central place managing the execution of all configured LLM models and MCP servers.
    """

    def __init__(
        self,
        llm_models: Dict[str, ModelConfig],
        mcp_servers: Dict[str, Dict],
        default_model: Optional[ModelConfig] = None,
    ):
        self.llm_models = llm_models
        self.default_model = default_model
        self.mcp_caller = MCPServersCaller(mcp_servers=mcp_servers)
        self.llm_engines: Dict[str, LLMInferenceEngine] = {}

    def get_mcp_caller(self) -> MCPServersCaller:
        return self.mcp_caller

    def _select_llm_engine(self, model: str = None) -> LLMInferenceEngine:
        _model = self._get_llm_model(model)
        if not _model:
            raise RuntimeError(f"No model named {model}.")
        if _model.name in self.llm_engines:
            llm_engine = self.llm_engines[_model.name]
        else:
            llm_engine = create_llm_engine(_model)
            self.llm_engines[_model.name] = llm_engine
        return llm_engine

    def _get_llm_model(self, model: str | None) -> ModelConfig:
        if model:
            _model = self.llm_models.get(model)
        else:
            _model = self.default_model
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
        max_iterations: int = 10,
        confirm_tool_exec: bool = False,
        model: str = None,
        system_message_override: Optional[Callable[[], str]] = None,
        filter_tools: Optional[
            Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]
        ] = None,
        enhance_prompt: Optional[
            Callable[[str, MCPServersCaller], Awaitable[list[BaseMessage]]]
        ] = None,
        tool_error_check: Optional[
            Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]
        ] = None,
        skill_sets: List[str] = None,
        template: Optional[str] = None,
        params: dict = None,
        options: Dict[str, Any] = None,
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
                system_message_override,
                filter_tools,
                enhance_prompt,
                tool_error_check,
                skill_sets,
                template,
                params,
                options,
            )
        )

    async def stream_user_input(
        self,
        user_input: str,
        images: List[str] = None,
        audio_files: List[str] = None,
        video_files: List[str] = None,
        files: List[str] = None,  # General files for tools like code interpreter
        max_iterations: int = 10,
        confirm_tool_exec: bool = False,
        model: str = None,
        system_message_override: Optional[Callable[[], str]] = None,
        filter_tools: Optional[
            Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]
        ] = None,
        enhance_prompt: Optional[
            Callable[[str, MCPServersCaller], Awaitable[list[BaseMessage]]]
        ] = None,
        tool_error_check: Optional[
            Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]
        ] = None,
        skill_sets: List[str] = None,
        template: Optional[str] = None,
        params: dict = None,
        options: Dict[str, Any] = None,
    ) -> AsyncIterator[BaseMessage]:
        """
        Stream user input through the LLM, handling tool calls if needed.
        Args:
            user_input (str): The user input.
            images (List[str]): List of image file paths to send with the message.
            audio_files (List[str]): List of audio file paths to send with the message.
            video_files (List[str]): List of video file paths to send with the message.
            files (List[str]): List of general file paths to upload for tools like code interpreter.
            max_iterations (int): The maximum number of iterations.
            confirm_tool_exec(bool): Ask for confirmation on tool execution.
            model (str): The model to use, the default one will be used if not specified.
            system_message_override: The system message override function.
            filter_tools: The function that filters tool calls.
            enhance_prompt: The function that enhances the prompt. This works alongside the default function
                           that reads Cliver.md for context.
            tool_error_check: The function that checks tool errors. The returned string will be the tool error message
                              sent back to LLM if the first returned value is True.
            skill_sets: List of skill set names to apply.
            template: Template name to apply.
            params: Parameters for skill sets and templates.
            options: Dictionary of additional options to override LLM configurations.
        """

        (
            llm_engine,
            mcp_tools,
            messages,
            skill_set_tools,
        ) = await self._prepare_messages_and_tools(
            enhance_prompt,
            filter_tools,
            model,
            system_message_override,
            user_input,
            images,
            audio_files,
            video_files,
            files,
            skill_sets,
            template,
            params,
        )

        # Since we've enhanced the infer and stream methods to handle multimedia,
        # we can always use the regular stream method
        async for chunk in self._stream_messages(
            llm_engine,
            model,
            messages,
            max_iterations,
            0,
            mcp_tools,
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
        system_message_override,
        user_input,
        images=None,
        audio_files=None,
        video_files=None,
        files=None,  # General files for tools like code interpreter
        skill_sets=None,
        template=None,
        params=None,
    ):
        logger.info(f"_prepare_messages_and_tools called with files: {files}")

        # Check file upload capability early, before any processing
        if files:
            llm_engine = self._select_llm_engine(model)
            if hasattr(llm_engine, 'config'):
                capabilities = llm_engine.config.get_capabilities()
                from cliver.model_capabilities import ModelCapability
                if ModelCapability.FILE_UPLOAD not in capabilities:
                    logger.info("File upload is not supported for this model. Will use content embedding as fallback.")

        llm_engine = self._select_llm_engine(model)
        logger.info(f"Selected LLM engine: {type(llm_engine)}")
        # Add system message to instruct the LLM about tool usage
        system_message = llm_engine.system_message()
        if system_message_override:
            system_message = system_message_override()

        # Create initial messages with system message
        messages: list[BaseMessage] = [
            SystemMessage(content=system_message),
        ]
        # Always apply the default enhancement function to get context from Cliver.md
        default_enhanced_messages = await default_enhance_prompt(
            user_input, self.mcp_caller
        )
        if default_enhanced_messages and len(default_enhanced_messages) > 0:
            messages.extend(default_enhanced_messages)

        mcp_tools = await self.mcp_caller.get_mcp_tools()
        if filter_tools:
            mcp_tools = await filter_tools(user_input, mcp_tools)
        if not mcp_tools:
            mcp_tools = []

        # Apply skill sets and templates if provided
        skill_set_tools = []
        if skill_sets or template:
            messages, skill_set_tools = apply_skill_sets_and_template(
                user_input, messages, skill_sets, template, params
            )

        # Add skill set tools to mcp_tools if any
        if skill_set_tools:
            # TODO: Convert skill_set_tools to BaseTool objects
            # For now, we'll just log that we have skill set tools
            logger.info(f"Skill set tools to include: {skill_set_tools}")

        # Apply user-provided enhancement function if provided
        if enhance_prompt:
            user_enhanced_messages = await enhance_prompt(user_input, self.mcp_caller)
            messages.extend(user_enhanced_messages)

        # Load media files if provided
        media_content = []

        # Load image files
        logger.info(f"loading images: {images}")
        if images:
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
                logger.info("File upload is not supported for this model. Will embed file contents in the prompt instead.")
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
                        if hasattr(llm_engine, 'upload_file'):
                            file_id = llm_engine.upload_file(file_path)
                            if file_id:
                                uploaded_file_ids.append(file_id)
                                logger.info(f"Uploaded file {file_path} with ID {file_id}")
                            else:
                                logger.warning(f"Failed to upload file {file_path}")
                        else:
                            logger.info(f"LLM engine doesn't support file uploads, skipping {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not upload file {file_path}: {e}. Please check the configuration if the capability is enabled.")
                logger.info(f"Completed file uploads. Uploaded file IDs: {uploaded_file_ids}")

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
                        "text": f"\n\nUploaded files for reference: {', '.join(uploaded_file_ids)}"
                    }
                )

            # Add embedded file contents for models that don't support file uploads
            if embedded_files_content:
                content_parts.append({
                    "type": "text",
                    "text": "\n\nThe following files have been provided for context:"
                })
                for file_path, file_content in embedded_files_content:
                    content_parts.append({
                        "type": "text",
                        "text": f"\n\nFile: {file_path}\nContent:\n```\n{file_content}\n```"
                    })

            human_message = HumanMessage(content=content_parts)
            messages.append(human_message)
        else:
            # Add the user input as a regular message
            messages.append(HumanMessage(content=user_input))

        return llm_engine, mcp_tools, messages, skill_set_tools

    # This is the method that can be used out of box
    async def process_user_input(
        self,
        user_input: str,
        images: List[str] = None,
        audio_files: List[str] = None,
        video_files: List[str] = None,
        files: List[str] = None,  # General files for tools like code interpreter
        max_iterations: int = 10,
        confirm_tool_exec: bool = False,
        model: str = None,
        system_message_override: Optional[Callable[[], str]] = None,
        filter_tools: Optional[
            Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]
        ] = None,
        enhance_prompt: Optional[
            Callable[[str, MCPServersCaller], Awaitable[list[BaseMessage]]]
        ] = None,
        tool_error_check: Optional[
            Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]
        ] = None,
        skill_sets: List[str] = None,
        template: Optional[str] = None,
        params: dict = None,
        options: Dict[str, Any] = None,
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
            confirm_tool_exec(bool): Ask for confirmation on tool execution.
            model (str): The model to use, the default one will be used if not specified.
            system_message_override: The system message override function.
            filter_tools: The function that filters tool calls.
            enhance_prompt: The function that enhances the prompt. This works alongside the default function
                           that reads Cliver.md for context.
            tool_error_check: The function that checks tool errors. The returned string will be the tool error message
                              sent back to LLM if the first returned value is True.
            skill_sets: List of skill set names to apply.
            template: Template name to apply.
            params: Parameters for skill sets and templates.
        """

        (
            llm_engine,
            mcp_tools,
            messages,
            skill_set_tools,
        ) = await self._prepare_messages_and_tools(
            enhance_prompt,
            filter_tools,
            model,
            system_message_override,
            user_input,
            images,
            audio_files,
            video_files,
            files,
            skill_sets,
            template,
            params,
        )

        # Since we've enhanced the infer and stream methods to handle multimedia,
        # we can always use the regular infer method
        return await self._process_messages(
            llm_engine,
            model,
            messages,
            max_iterations,
            0,
            mcp_tools,
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
        confirm_tool_exec: bool,
        tool_error_check: Optional[
            Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]
        ] = None,
        infer_method: Optional[Callable] = None,
        options: Dict[str, Any] = None,
    ) -> BaseMessage:
        """Handle processing messages with tool calling using a while loop."""
        iteration = current_iteration

        # Use custom inference method if provided, otherwise use default
        if infer_method is None:
            infer_method = llm_engine.infer

        while iteration < max_iterations:
            # Get response from LLM
            response = await infer_method(messages, mcp_tools, options=options)
            logger.debug(f"LLM response: {response}")
            # Handle different response types
            tool_calls = llm_engine.parse_tool_calls(response, model)
            if tool_calls:
                # as long as there is tool execution, the result will be sent back unless a fatal error happens
                stop, result = await self._execute_tool_calls(
                    tool_calls,
                    messages,
                    confirm_tool_exec,
                    tool_error_check,
                )
                if stop:
                    return AIMessage(content=result)
                else:
                    # If sent is False, continue processing messages
                    iteration += 1
                    continue
            # If no tool calls, return the response
            return response

        return AIMessage(
            content="Reached maximum number of iterations without a final answer."
        )

    # returns a tuple to indicate if there is error occurs which indicates stop and a string message
    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        messages: List[BaseMessage],
        confirm_tool_exec: bool,
        tool_error_check: Optional[
            Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]
        ],
    ) -> Tuple[bool, str | None]:
        """Execute tool calls and return a Tuple with sent bool and result."""
        try:
            for tool_call in tool_calls:
                mcp_server = tool_call["mcp_server"]
                tool_name = tool_call["tool_name"]
                args = tool_call["args"]
                tool_call_id = tool_call["tool_call_id"]
                # default we don't care about confirmation and just run the tool
                proceed = True
                if confirm_tool_exec:
                    proceed = _confirm_tool_execution(
                        f"This will execute tool: {tool_name} from mcp server: {mcp_server}"
                    )
                if not proceed:
                    return True, f"Stopped at tool execution: {tool_call}"

                # Append the tool execution to messages using AIMessage with tool_calls
                tool_execution_message = AIMessage(
                    content="",  # Empty content as this is a tool call
                    tool_calls=[
                        {
                            "name": f"{mcp_server}#{tool_name}",
                            "args": args,
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                )
                messages.append(tool_execution_message)

                mcp_tool_result = await retry_with_confirmation_async(
                    self.mcp_caller.call_mcp_server_tool,
                    mcp_server,
                    tool_name,
                    args,
                    confirm_on_retry=False,
                )
                # Format the tool result properly for ToolMessage
                if isinstance(mcp_tool_result, list) and len(mcp_tool_result) > 0:
                    first_result = mcp_tool_result[0]
                    if isinstance(first_result, dict) and "text" in first_result:
                        tool_result_content = first_result["text"]
                    else:
                        tool_result_content = str(mcp_tool_result)
                else:
                    tool_result_content = str(mcp_tool_result)
                messages.append(
                    ToolMessage(content=tool_result_content, tool_call_id=tool_call_id)
                )
                if not tool_error_check:
                    tool_error_check = _tool_error_check_internal
                error, tool_error_message = tool_error_check(tool_name, mcp_tool_result)
                if error:
                    messages.append(AIMessage(content=tool_error_message))
                    return error, tool_error_message
            # all good, messages have been updated, go on with next iteration
            return False, "Tool calls executed successfully"
        except Exception as e:
            logger.error(f"Error processing tool call: {str(e)}", exc_info=True)
            return True, f"Error processing tool call: {str(e)}"

    async def _stream_messages(
        self,
        llm_engine: LLMInferenceEngine,
        model: str,
        messages: List[BaseMessage],
        max_iterations: int,
        current_iteration: int,
        mcp_tools: list[BaseTool],
        confirm_tool_exec: bool,
        tool_error_check: Optional[
            Callable[[str, list[Dict[str, Any]]], Tuple[bool, str | None]]
        ],
        stream_method: Optional[Callable] = None,
        options: Dict[str, Any] = None,
    ) -> AsyncIterator[BaseMessage]:
        """Handle streaming messages with tool calling."""
        iteration = current_iteration
        tool_call_indicators = [
            'tool_calls"',
            '"tool_calls"',
            '{"tool_calls"',
            "'tool_calls'",
        ]

        # Use custom streaming method if provided, otherwise use default
        if stream_method is None:
            stream_method = llm_engine.stream

        while iteration < max_iterations:
            # Stream response from LLM
            accumulated_content = ""
            tool_call_buffer = ""  # Buffer specifically for tool call detection
            last_yield_time = 0

            async for chunk in stream_method(messages, mcp_tools, options=options):
                current_time = time.time()

                # Accumulate content from chunks to handle incomplete tool calls
                chunk_content = ""
                if hasattr(chunk, "content") and chunk.content:
                    chunk_content = str(chunk.content)
                    accumulated_content += chunk_content

                # Check if this chunk contains tool calls
                tool_calls = None

                # First check if the chunk directly has tool_calls attribute
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    try:
                        logger.debug(f"Streaming chunk with tool calls: {chunk}")
                        tool_calls = llm_engine.parse_tool_calls(chunk, model)
                        if tool_calls is None:
                            continue
                    except Exception as e:
                        logger.error(
                            f"Error processing tool call in streaming: {str(e)}",
                            exc_info=True,
                        )
                        # Continue processing other chunks
                        continue
                else:
                    # Add chunk content to tool call buffer for detection
                    # Limit buffer size to prevent memory issues
                    tool_call_buffer = (tool_call_buffer + chunk_content)[-200:]
                    if any(
                        tool_call_buffer.strip() in indicator
                        for indicator in tool_call_indicators
                    ):
                        # it might be the case the first chunks are very small which we have no idea of if it is a tool call or not.
                        continue

                    # Check if buffer suggests this is a tool call
                    # More robust detection than just checking for a substring
                    if any(
                        indicator in tool_call_buffer
                        for indicator in tool_call_indicators
                    ):
                        try:
                            # Create a temporary message with accumulated content for parsing
                            temp_message = AIMessage(content=accumulated_content)
                            tool_calls = llm_engine.parse_tool_calls(
                                temp_message, model
                            )
                            if tool_calls is None:
                                continue
                        except Exception as e:
                            # If parsing fails, it might be an incomplete tool call
                            # Continue accumulating content
                            logger.error(
                                f"Incomplete tool call detected, continuing accumulation: {str(e)}"
                            )
                            continue

                if tool_calls:
                    # Reset buffers as we found complete tool calls
                    accumulated_content = ""
                    tool_call_buffer = ""
                    # Execute tool calls (this method is async, so we need to await it)
                    stop, result = await self._execute_tool_calls(
                        tool_calls, messages, confirm_tool_exec, tool_error_check
                    )
                    if stop:
                        # If we need to stop, yield the result as a message
                        if result:
                            yield AIMessage(content=result)
                        return
                    else:
                        # Continue processing with the updated messages
                        iteration += 1
                        break  # Break inner loop to continue with next iteration
                else:
                    # Only yield the chunk if we're confident it's not part of a tool call
                    # Check that it doesn't contain tool call indicators
                    # But also check if we've accumulated too much content without finding tool calls
                    # Or if enough time has passed since last yield (to prevent hanging)
                    safe_to_yield = (
                        not any(
                            indicator in accumulated_content
                            for indicator in tool_call_indicators
                        )
                        or len(accumulated_content)
                        > 1000  # If we've accumulated a lot, it's probably not a tool call
                        or (current_time - last_yield_time)
                        > 2.0  # If 2 seconds have passed, yield what we have
                    )

                    if safe_to_yield and chunk_content:
                        yield chunk
                        last_yield_time = current_time
                    # Continue to next chunk without yielding if we're still unsure

            else:
                # If we completed the async for loop without breaking, we're done
                return

        # If we've reached max iterations
        yield AIMessage(
            content="Reached maximum number of iterations without a final answer."
        )


def _tool_error_check_internal(
    tool_name: str, mcp_tool_result: list[Dict[str, Any]]
) -> Tuple[bool, str | None]:
    if any("error" in r for r in mcp_tool_result):
        return (
            True,
            f"Error calling tool {tool_name}: {mcp_tool_result[0].get('error')}, you may need to check the tool arguments and run it again.",
        )
    return False, None


def _confirm_tool_execution(prompt="Are you sure? (y/n): ") -> bool:
    while True:
        response = input(prompt).strip().lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
