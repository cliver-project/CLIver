"""DeepSeek-specific inference engine.

Handles DeepSeek API quirks that differ from standard OpenAI-compatible APIs:
- reasoning_content field in responses (for DeepSeek-R1 / deepseek-reasoner)
- Content array flattening (DeepSeek requires content as string, not array)
- reasoning_content must NOT be included in conversation history input
"""

import logging
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

from cliver.config import ModelConfig
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine

logger = logging.getLogger(__name__)


class DeepSeekInferenceEngine(OpenAICompatibleInferenceEngine):
    def __init__(self, config: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"):
        super().__init__(config, user_agent=user_agent, agent_name=agent_name)

    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Convert messages to DeepSeek format.

        DeepSeek requires content to be a plain string, not an array of
        content parts. This method flattens multipart content into a single
        string, then delegates media handling to the parent class.
        """
        converted = []
        for message in messages:
            if isinstance(message, HumanMessage) and isinstance(message.content, list):
                # Flatten content parts into a single string
                text_parts = []
                for part in message.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            text_parts.append("[Image attached]")
                        else:
                            text_parts.append(str(part))
                    elif isinstance(part, str):
                        text_parts.append(part)
                flattened = HumanMessage(content="\n\n".join(text_parts))
                converted.append(flattened)
            else:
                converted.append(message)

        return converted
