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
                _llm = self.llm.bind_tools(tools)
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
