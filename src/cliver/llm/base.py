from abc import ABC, abstractmethod
from cliver.config import ModelConfig
from typing import List, Optional, AsyncIterator
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool


class LLMInferenceEngine(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config or {}

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

After you make a tool call, you will receive the result. Use that information to formulate your final answer.

If you have all the information needed to answer directly without using any tools, provide a text response.

Important:
1. Only use the exact tool names provided to you
2. Respond ONLY with the JSON format when calling tools
3. Do not include any other text when making tool calls
4. Wait for the tool results before providing your final answer
"""
