from abc import ABC, abstractmethod
from cliver.config import ModelConfig
from typing import List, Optional
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

class LLMInferenceEngine(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config or {}

    # This method focus on the real LLM inference only.
    @abstractmethod
    async def infer(self, messages: List[BaseMessage], tools: Optional[list[BaseTool]]) -> BaseMessage:
        pass

    def system_message(self) -> str:
        """
        This method can be overridden
        """
        return """
        You are an AI assistant that can use tools to help answer questions.

If you need to call a tools to get more information from client side, always extract MCP tool calls and format them **only** as a function call to `tool_calls`.
- Gather from user: tool_name, arguments.
- When ready, respond only with JSON in the form:
  "tool_calls": [
    {
      "name": "<tool_name>",
      "args": {...},
      "id": "<tool_call_id>",
      "type": "tool_call"
    }
  ]
- Do **not** add any text outside the JSON.

If you have all the information needed to answer directly, provide a text response.
"""