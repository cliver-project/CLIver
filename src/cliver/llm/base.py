import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional, Any
import uuid
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessageChunk, AIMessageChunk, AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

from cliver.config import ModelConfig
from cliver.llm.errors import get_friendly_error_message
from cliver.llm.llm_utils import parse_tool_calls_from_content
from cliver.media import MediaContent
from cliver.model_capabilities import ModelCapability

logger = logging.getLogger(__name__)

class LLMInferenceEngine(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config or {}
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
            friendly_error_msg = [get_friendly_error_message(e, "LLM inference"), f"\tmodel: {self.config.name}"]
            return AIMessage(content=f"Error: {"\n".join(friendly_error_msg)}", additional_kwargs={"type": "error"})

    async def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[list[BaseTool]],
        ** kwargs: Any,
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
            friendly_error_msg = [get_friendly_error_message(e, "OpenAI inference"), f"\tmodel: {self.config.name}"]
            # noinspection PyArgumentList
            yield AIMessageChunk(content=f"Error: {"\n".join(friendly_error_msg)}", additional_kwargs={"type": "error"})

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
        This method can be overridden
        """
        # Check if the model supports thinking mode based on its name
        # Models known to support thinking mode include DeepSeek-r1 and Qwen models
        supports_thinking = self.supports_capability(ModelCapability.THINK_MODE)
        thinking_part1 = ""
        thinking_part2 = ""
        if supports_thinking:
            thinking_part1 = """
You have a thinking mode capability where you can reason through problems before providing your final answer.

When solving complex problems, you can use thinking sections to reason through the problem:
<thinking>
[Your detailed reasoning process here]
</thinking>
"""
            thinking_part2 = """
IMPORTANT INSTRUCTIONS FOR THINKING MODE:
1. Use <thinking>...</thinking> sections to show your reasoning process
2. Surround your thinking content with '<thinking>' and '</thinking>' tags
3. Place thinking sections at the beginning of your response when complex reasoning is needed
"""

        return f"""
You are an AI assistant that can use tools to help answer questions.
{thinking_part1}

Available tools will be provided to you. When you need to use a tool, you MUST use the exact tool name provided.
To use a tool, respond in the proper format that allows the system to process your tool call.

For models that support structured tool calling (like OpenAI, Anthropic, etc.), use the formal tool calling mechanism.
For models that don't support structured tool calling, respond with JSON in this exact format when making tool calls:
{{
  "tool_calls": [
    {{
      "name": "exact_tool_name",
      "args": {{
        "argument_name": "argument_value"
      }},
      "id": "unique_identifier_for_this_call",
      "type": "tool_call"
    }}
  ]
}}

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. Only use the exact tool names provided to you
2. When calling tools, your response should contain ONLY the tool call format - no additional text unless providing a final answer
3. Generate a unique ID for each tool call using standard UUID format if not using formal tool calling
4. Always include the "type": "tool_call" field in JSON format
5. Ensure your JSON is properly formatted and parsable
6. The tool_calls array must be a valid JSON array
7. After making tool calls, wait to receive the results before proceeding
8. Do not embed tool calls in markdown code blocks unless the system specifically handles them
9. If providing a final answer without using tools, respond with normal text

{thinking_part2}

After you make a tool call, you will receive the result. You may need to make additional tool calls based on the results until you have enough information to provide your final answer. The process can involve multiple rounds of tool calls.
If you have all the information needed to answer directly without using any tools, provide a text response.
"""

    async def _reconstruct_llm(self, _llm: BaseChatModel, tools: Optional[list[BaseTool]]) -> BaseChatModel:
        if tools and len(tools) > 0:
            # Check if the model supports tool calling
            capabilities = self.config.get_capabilities()
            if ModelCapability.TOOL_CALLING in capabilities:
                _llm = _llm.bind_tools(tools, strict=True)
        return _llm
