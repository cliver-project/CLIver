import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

from cliver.config import ModelConfig
from cliver.llm.errors import get_friendly_error_message
from cliver.llm.llm_utils import parse_tool_calls_from_content
from cliver.media import MediaContent
from cliver.model_capabilities import ModelCapability

logger = logging.getLogger(__name__)


class LLMInferenceEngine(ABC):
    def __init__(self, config: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"):
        self.config = config or {}
        self.user_agent = user_agent
        self.agent_name = agent_name
        self.llm = None

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if the model supports a specific capability."""
        return self.config.get_model_capabilities().supports(capability)

    # This method focus on the real LLM inference only.
    async def infer(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> BaseMessage:
        try:
            # Convert messages to LLM engine format if needed
            converted_messages = self.convert_messages_to_engine_specific(messages)
            _llm = await self._reconstruct_llm(self.llm, tools)
            response = await _llm.ainvoke(converted_messages, **kwargs)
            return response
        except Exception as e:
            logger.debug(f"Error in infer: {str(e)}", exc_info=True)
            friendly_error_msg = [
                get_friendly_error_message(e, "LLM inference"),
                f"\tmodel: {self.config.name}",
            ]
            return AIMessage(
                content=f"Error: {'\n'.join(friendly_error_msg)}",
                additional_kwargs={"type": "error"},
            )

    async def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        """Stream responses from the LLM."""
        try:
            # Convert messages to OpenAI multi-media format if needed
            converted_messages = self.convert_messages_to_engine_specific(messages)
            _llm = await self._reconstruct_llm(self.llm, tools)
            # noinspection PyTypeChecker
            async for chunk in _llm.astream(input=converted_messages, **kwargs):
                yield chunk
        except Exception as e:
            logger.debug(f"Error in OpenAI stream: {str(e)}", exc_info=True)
            friendly_error_msg = [
                get_friendly_error_message(e, "OpenAI inference"),
                f"\tmodel: {self.config.name}",
            ]
            # noinspection PyArgumentList
            yield AIMessageChunk(
                content=f"Error: {'\n'.join(friendly_error_msg)}",
                additional_kwargs={"type": "error"},
            )

    @abstractmethod
    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        pass

    def extract_media_from_response(self, response: BaseMessage) -> List[MediaContent]:
        """
        Extract media content from LLM response.

        This method should be overridden by specific LLM engines to handle
        their specific response formats for multimedia content.

        Args:
            response: BaseMessage response from the LLM

        Returns:
            List of MediaContent objects extracted from the response
        """
        # Default implementation returns empty list
        # Specific engines should override this method
        return []

    def parse_tool_calls(self, response: BaseMessage, model: str) -> list[dict] | None:
        """Parse the tool calls from the response from the LLM."""
        if response is None:
            return None
        # tool_calls may differ based on different model and provider
        # each own engine implementation can do their own manipulate to the ones cliver knows
        tool_calls = parse_tool_calls_from_content(response)
        if tool_calls is None:
            return None
        return self.convert_tool_calls_for_execute(tool_calls)

    def convert_tool_calls_for_execute(self, tool_calls: list[dict]) -> list[dict] | None:
        logger.debug("tool_calls to convert into execution: %s", tool_calls)
        tools_to_call = []
        for tool_call in tool_calls:
            tool_to_call = {}
            mcp_server_name = ""
            tool_name: str = tool_call.get("name")
            the_tool_name = tool_name
            if "#" in tool_name:
                s_array = tool_name.split("#")
                mcp_server_name = s_array[0]
                the_tool_name = s_array[1]
            args = tool_call.get("args")
            tool_call_id = tool_call.get("id")
            # Ensure we have a valid tool_call_id for OpenAI compatibility
            if not tool_call_id:
                tool_call_id = str(uuid.uuid4())
            tool_to_call["tool_call_id"] = tool_call_id
            tool_to_call["tool_name"] = the_tool_name
            tool_to_call["mcp_server"] = mcp_server_name
            tool_to_call["args"] = args
            tools_to_call.append(tool_to_call)
        return tools_to_call

    def system_message(self) -> str:
        """
        Build the builtin system prompt for CLIver.

        This method can be overridden by engine subclasses.
        User-provided system messages are appended separately by TaskExecutor.
        """
        sections = [self._section_identity(self.agent_name)]
        sections.append(self._section_tool_usage())
        sections.append(self._section_interaction_guidelines())
        sections.append(self._section_response_format())

        return "\n\n".join(sections)

    # -- System prompt sections ------------------------------------------------

    @staticmethod
    def _section_identity(agent_name: str) -> str:
        return (
            "# Identity\n\n"
            f"You are **{agent_name}**, a general-purpose AI agent. "
            "You help users accomplish a wide variety of tasks — answering questions, "
            "searching the web, reading and writing files, running commands, managing containers, "
            "and anything else the user asks for.\n\n"
            "You can operate in different environments: command-line interfaces, "
            "embedded applications, or as a backend service. "
            "Adapt your tone, depth, and approach to whatever the user needs."
        )

    @staticmethod
    def _section_tool_usage() -> str:
        return (
            "# Tool Usage\n\n"
            "You have access to tools that extend your capabilities. "
            "The available tools, their parameter schemas (names, types, required/optional), "
            "and descriptions are provided alongside this message.\n\n"
            "## How to call tools\n\n"
            "- Use the **structured tool-calling mechanism** provided by the model API.\n"
            "- Use the **exact tool name** as given — do not invent or guess tool names.\n"
            "- Supply arguments that match the parameter schema (correct types, required fields).\n"
            "- You may call **multiple tools** in a single response when the calls are independent.\n\n"
            "## Fallback format (for models without structured tool calling)\n\n"
            "If the model does not support structured tool calling, respond with **only** "
            "the following JSON (no surrounding text):\n\n"
            "```json\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "name": "exact_tool_name",\n'
            '      "args": { "param": "value" },\n'
            '      "id": "unique_uuid",\n'
            '      "type": "tool_call"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "## Iterative tool use\n\n"
            "After each tool call you will receive the result. You may make **additional tool calls** "
            "based on the results until you have enough information to provide a final answer. "
            "This process can involve multiple rounds.\n\n"
            "If you already have enough information, respond directly without calling any tools."
        )

    @staticmethod
    def _section_interaction_guidelines() -> str:
        return (
            "# Interaction Guidelines\n\n"
            "## Asking the user\n\n"
            "Use the `ask_user_question` tool when you need to:\n"
            "- Clarify ambiguous instructions\n"
            "- Confirm before destructive or irreversible actions\n"
            "- Choose between multiple valid approaches\n"
            "- Gather missing information that you cannot determine on your own\n\n"
            "Do **not** ask for confirmation on routine, safe, or clearly specified tasks.\n\n"
            "## Task planning\n\n"
            "For complex tasks that involve **3 or more steps**, use the `todo_write` tool "
            "to create a structured work breakdown before starting. "
            "Update the todo list as you progress — mark items `in_progress` when you begin "
            "and `completed` when done.\n\n"
            "## Error handling\n\n"
            "If a tool call fails, analyse the error and try an alternative approach "
            "rather than repeating the same call. "
            "If you are unable to resolve the issue, explain what went wrong to the user."
        )

    @staticmethod
    def _section_response_format() -> str:
        return (
            "# Response Format\n\n"
            "- Respond in **Markdown** format for readability.\n"
            "- Be concise and direct; avoid unnecessary preamble.\n"
            "- When presenting structured data, use tables, lists, or code blocks as appropriate.\n"
            "- When a tool call is needed, respond with **only** the tool call — "
            "do not mix tool calls with prose in the same response."
        )

    async def _reconstruct_llm(self, _llm: BaseChatModel, tools: Optional[list[BaseTool]]) -> BaseChatModel:
        if tools and len(tools) > 0:
            capabilities = self.config.get_capabilities()
            if ModelCapability.TOOL_CALLING in capabilities:
                # Only use strict=True for providers that fully support OpenAI's
                # strict function calling (native OpenAI). Most OpenAI-compatible
                # providers (DeepSeek, Qwen, GLM, vLLM) don't support it.
                use_strict = ModelCapability.FUNCTION_CALLING in capabilities
                _llm = _llm.bind_tools(tools, strict=use_strict)
        return _llm
