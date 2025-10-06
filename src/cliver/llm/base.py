import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

from cliver.config import ModelConfig
from cliver.media import MediaContent
from cliver.model_capabilities import ModelCapability

logger = logging.getLogger(__name__)


class LLMInferenceEngine(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config or {}

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if the model supports a specific capability."""
        capabilities = self.config.get_capabilities()
        return capability in capabilities

    # This method focus on the real LLM inference only.
    @abstractmethod
    async def infer(
        self, messages: List[BaseMessage], tools: Optional[list[BaseTool]]
    ) -> BaseMessage:
        pass

    async def stream(
        self, messages: List[BaseMessage], tools: Optional[list[BaseTool]]
    ) -> AsyncIterator[BaseMessage]:
        """Stream responses from the LLM."""
        # Default implementation falls back to regular inference
        response = await self.infer(messages, tools)
        yield response

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
        tool_calls = self._parse_tool_calls_from_content(response, model)
        if tool_calls is None:
            return None
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
                import uuid

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
        return """
You are an AI assistant that can use tools to help answer questions.

Available tools will be provided to you. When you need to use a tool, you MUST use the exact tool name provided.

To use a tool, respond ONLY with a JSON object in this exact format:
{
  "tool_calls": [
    {
      "name": "exact_tool_name",
      "args": {
        "argument_name": "argument_value"
      },
      "id": "unique_identifier_for_this_call",
      "type": "tool_call"
    }
  ]
}

After you make a tool call, you will receive the result. You may need to make additional tool calls based on the results until you have enough information to provide your final answer. The process can involve multiple rounds of tool calls.

If you have all the information needed to answer directly without using any tools, provide a text response.

Important:
1. Only use the exact tool names provided to you
2. Respond ONLY with the JSON format when calling tools
3. Do not include any other text when making tool calls
4. Wait for the tool results before providing your final answer
5. You can make multiple rounds of tool calls if needed - after receiving results, you can make another tool call or provide your final answer
6. The tool_calls should be returned as a strict JSON format that can be accessed via response.tool_calls, not embedded in the response content
"""

    def _parse_tool_calls_from_content(
        self, response: BaseMessage, model: str
    ) -> Optional[list[dict]]:
        """Parse tool calls from response content when LLM doesn't properly use tool binding."""
        if response is None:
            return None
        if response.tool_calls:
            return response.tool_calls
        if hasattr(response, "content") and response.content:
            if type(response.content) == dict and dict(response.content)["tool_calls"]:
                content_dict = dict(response.content)
                return content_dict.get("tool_calls", [])
        if (
            hasattr(response, "content")
            and response.content
            and '"tool_calls"' in str(response.content)
        ):
            try:
                import re

                import json_repair

                content_str = str(response.content)
                # Look for tool_calls pattern in the content
                # This pattern matches the JSON structure we expect
                pattern = r'\{[^{]*"tool_calls":\s*\[[^\]]*\][^\}]*\}'
                match = re.search(pattern, content_str, re.DOTALL)

                if match:
                    # Extract the complete JSON object containing tool_calls
                    tool_calls_section = match.group(0)
                    parsed = json_repair.loads(tool_calls_section)
                    tool_calls = parsed.get("tool_calls", [])
                    return tool_calls

                # If the above pattern doesn't work, try to find just the tool_calls array
                pattern = r'"tool_calls":\s*(\[[^\]]*\])'
                match = re.search(pattern, content_str, re.DOTALL)

                if match:
                    # Extract just the tool_calls array
                    tool_calls_array = match.group(1)
                    tool_calls = json_repair.loads(tool_calls_array)
                    return tool_calls
            except Exception as e:
                # If parsing fails, return None
                logger.error(f"Error parsing tool calls: {str(e)}", exc_info=True)
                return None
        return None
