"""DeepSeek-specific inference engine.

Uses langchain-deepseek's ChatDeepSeek directly — no subclassing.

Handles DeepSeek API quirks via convert_messages_to_engine_specific():
- Content array flattening (DeepSeek requires content as plain string)
- Strips reasoning_content from conversation history so DeepSeek's API
  doesn't require it on every assistant message. The model still uses
  thinking mode for the current response; prior reasoning is not needed.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine

logger = logging.getLogger(__name__)


class DeepSeekInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"):
        super().__init__(config, user_agent=user_agent, agent_name=agent_name)
        self.options = {}
        if self.config and self.config.options:
            self.options = self.config.options.model_dump(exclude_unset=True)

        resolved_api_key = self.config.get_api_key()
        resolved_url = self.config.get_resolved_url()
        default_headers = {"User-Agent": user_agent} if user_agent else {}

        llm_kwargs: Dict[str, Any] = {
            "model": self.config.api_model_name,
            "api_key": resolved_api_key,
            "default_headers": default_headers,
        }

        if resolved_url:
            llm_kwargs["api_base"] = resolved_url

        for key in ("temperature", "top_p", "max_tokens"):
            if key in self.options:
                llm_kwargs[key] = self.options[key]

        extra = {k: v for k, v in self.options.items() if k not in llm_kwargs}
        if extra:
            llm_kwargs["model_kwargs"] = extra

        self.llm = ChatDeepSeek(**llm_kwargs)

    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Convert messages for DeepSeek's API.

        - Flattens multipart HumanMessage content to plain strings
        - Strips reasoning_content from AIMessages to avoid the
          'reasoning_content must be passed back' error
        """
        converted = []
        for message in messages:
            if isinstance(message, HumanMessage) and isinstance(message.content, list):
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
                converted.append(HumanMessage(content="\n\n".join(text_parts)))
            elif isinstance(message, AIMessage) and message.additional_kwargs.get("reasoning_content"):
                cleaned_kwargs = {k: v for k, v in message.additional_kwargs.items() if k != "reasoning_content"}
                converted.append(
                    AIMessage(
                        content=message.content,
                        tool_calls=message.tool_calls,
                        additional_kwargs=cleaned_kwargs,
                    )
                )
            else:
                converted.append(message)
        return converted
