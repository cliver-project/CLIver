from typing import Optional, AsyncIterator

from cliver.config import ModelConfig
from cliver.model_capabilities import ModelCapability
from langchain_core.messages import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool
from cliver.llm.base import LLMInferenceEngine
from langchain_ollama import ChatOllama as Ollama


# Ollama inference engine
class OllamaLlamaInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.llm = Ollama(
            base_url=self.config.url,
            model=self.config.name_in_provider,
            **self.config.model_dump(),
        )

    async def infer(
        self, messages: list[BaseMessage], tools: Optional[list[BaseTool]]
    ) -> BaseMessage:
        try:
            _llm = self.llm
            if tools:
                # Check if the model supports tool calling
                capabilities = self.config.get_capabilities()
                if ModelCapability.TOOL_CALLING in capabilities:
                    _llm = self.llm.bind_tools(tools)
                else:
                    # Fallback to non-tool binding if not supported
                    pass
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
            # Check if the model supports tool calling
            capabilities = self.config.get_capabilities()
            if ModelCapability.TOOL_CALLING in capabilities:
                _llm = self.llm.bind_tools(tools)
            else:
                # Fallback to non-tool binding if not supported
                pass
        try:
            async for chunk in _llm.astream(messages):
                yield chunk
        except Exception as e:
            yield AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})
