"""Anthropic-compatible inference engine.

Uses langchain-anthropic's ChatAnthropic for consistent integration.
Handles Anthropic API specifics:
- System messages extracted automatically (top-level 'system' param)
- Thinking content via structured content blocks
- Tool calling via Anthropic's native tool format
- base_url support for Anthropic-compatible providers (e.g., MiniMax)
"""

import logging
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.media import MediaContent

logger = logging.getLogger(__name__)


class AnthropicInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"):
        super().__init__(config, user_agent=user_agent, agent_name=agent_name)
        self.options = {}
        if self.config and self.config.options:
            self.options = self.config.options.model_dump(exclude_unset=True)

        resolved_api_key = self.config.get_api_key()
        resolved_url = self.config.get_resolved_url()
        default_headers = {"User-Agent": user_agent} if user_agent else {}

        # ChatAnthropic constructor kwargs
        llm_kwargs: Dict[str, Any] = {
            "model": self.config.api_model_name,
            "anthropic_api_key": resolved_api_key,
            "default_headers": default_headers,
        }

        # base_url for Anthropic-compatible providers
        if resolved_url:
            llm_kwargs["anthropic_api_url"] = resolved_url

        # Map supported options
        for key in ("temperature", "top_p", "top_k", "max_tokens"):
            if key in self.options:
                llm_kwargs[key] = self.options[key]

        # Pass remaining options as model_kwargs
        extra = {k: v for k, v in self.options.items() if k not in llm_kwargs}
        if extra:
            llm_kwargs["model_kwargs"] = extra

        self.llm = ChatAnthropic(**llm_kwargs)

    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Convert messages for Anthropic.

        ChatAnthropic handles SystemMessage extraction and multimodal
        format conversion (image_url → Anthropic image blocks) natively.
        Messages pass through unchanged.
        """
        return list(messages)

    def extract_media_from_response(self, response: BaseMessage) -> List[MediaContent]:
        """Anthropic responses don't contain inline media."""
        return []
