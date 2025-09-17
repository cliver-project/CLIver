from typing import Optional, AsyncIterator

from cliver.config import ModelConfig
from langchain_core.messages import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool
from cliver.llm.base import LLMInferenceEngine
from langchain_openai import ChatOpenAI

# OpenAI compatible inference engine
class OpenAICompatibleInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Prepare the parameters for the ChatOpenAI constructor
        llm_params = {
            "model": self.config.name_in_provider or self.config.name,
            "base_url": self.config.url,
            "temperature": (
                self.config.options.temperature if self.config.options else 0.7
            ),
            "max_tokens": (
                self.config.options.max_tokens if self.config.options else 4096
            ),
        }

        # Add API key if provided
        if self.config.api_key:
            llm_params["api_key"] = self.config.api_key

        # Add any additional options from the config
        if self.config.options:
            for key, value in self.config.options.model_dump().items():
                # Skip temperature and max_tokens as they're already handled
                if key not in ["temperature", "max_tokens"]:
                    llm_params[key] = value

        self.llm = ChatOpenAI(**llm_params)

    async def infer(
        self, messages: list[BaseMessage], tools: Optional[list[BaseTool]]
    ) -> BaseMessage:
        try:
            _llm = self.llm
            if tools:
                _llm = self.llm.bind_tools(tools, strict=True)
            response = await _llm.ainvoke(messages)
            return response
        except Exception as e:
            return AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})

    async def stream(
        self, messages: list[BaseMessage], tools: Optional[list[BaseTool]]
    ) -> AsyncIterator[BaseMessage]:
        """Stream responses from the LLM."""
        _llm = self.llm
        if tools:
            _llm = self.llm.bind_tools(tools)
        try:
            async for chunk in _llm.astream(messages):
                yield chunk
        except Exception as e:
            yield AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})

    def system_message(self) -> str:
        """
        System message optimized for OpenAI-compatible models to ensure stable JSON tool calling format.
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

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. Only use the exact tool names provided to you
2. Respond ONLY with the JSON format when calling tools - no other text or explanation
3. Generate a unique ID for each tool call using standard UUID format
4. Always include the "type": "tool_call" field
5. Ensure your JSON is properly formatted and parsable
6. The tool_calls array must be a valid JSON array
7. Do not embed tool calls in markdown code blocks or any other formatting

After you make a tool call, you will receive the result. You may need to make additional tool calls based on the results until you have enough information to provide your final answer. The process can involve multiple rounds of tool calls.

If you have all the information needed to answer directly without using any tools, provide a text response.

Examples of CORRECT tool usage:
{"tool_calls": [{"name": "get_current_weather", "args": {"location": "New York"}, "id": "call_1234567890abcdef", "type": "tool_call"}]}

Examples of INCORRECT tool usage:
- Using markdown code blocks
- Adding explanatory text before/after JSON
- Missing required fields
- Improperly formatted JSON
"""