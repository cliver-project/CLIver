from typing import Optional

from cliver.config import ModelConfig
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
            **self.config.model_dump()
        )

    async def infer(self, messages: list[BaseMessage], tools: Optional[list[BaseTool]]) -> BaseMessage:
        try:
            _llm = self.llm
            if tools:
                _llm = self.llm.bind_tools(tools)
            response = await _llm.ainvoke(messages)
            return response
        except Exception as e:
            return AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})

