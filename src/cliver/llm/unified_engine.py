"""Unified LLM inference engine.

Uses LangChain's init_chat_model() for provider-agnostic ChatModel
construction.  Supports two API protocols:

    openai    — OpenAI-compatible REST/SSE API (OpenAI, DeepSeek, MiniMax,
                Ollama, Groq, xAI, and any other provider with an
                OpenAI-compatible endpoint).
    anthropic — Anthropic-native content-block API.

New providers are config-only: add a provider entry with the right
``api_url`` and a model entry with the right ``options`` — zero engine
changes needed.
"""

import json
import logging
from typing import Any, Callable, Dict, List

import langchain_openai.chat_models.base as _lc_openai_base
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.media_utils import data_url_to_media_content, extract_data_urls
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# reasoning_content round-trip support
# ---------------------------------------------------------------------------
# LangChain's ChatOpenAI ignores ``reasoning_content`` at every boundary:
#
#   API → LangChain (streaming):   _convert_delta_to_message_chunk
#   API → LangChain (non-streaming): _convert_dict_to_message
#   LangChain → API (both):          _convert_message_to_dict
#
# We patch all three once at module level.


# --- extraction (streaming): copy reasoning_content from delta into additional_kwargs ---

_original_convert_delta = _lc_openai_base._convert_delta_to_message_chunk


def _patched_convert_delta_to_message_chunk(_dict, default_class):
    chunk = _original_convert_delta(_dict, default_class)
    if isinstance(chunk, AIMessage) and isinstance(_dict, dict):
        rc = _dict.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            chunk.additional_kwargs["reasoning_content"] = rc
    return chunk


_lc_openai_base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk


# --- extraction (non-streaming): copy reasoning_content from response dict -----

_original_convert_dict = _lc_openai_base._convert_dict_to_message


def _patched_convert_dict_to_message(_dict):
    message = _original_convert_dict(_dict)
    if isinstance(message, AIMessage) and isinstance(_dict, dict):
        rc = _dict.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            message.additional_kwargs["reasoning_content"] = rc
    return message


_lc_openai_base._convert_dict_to_message = _patched_convert_dict_to_message


# --- injection: read reasoning_content from additional_kwargs into request dict ---

_original_convert_message_to_dict = _lc_openai_base._convert_message_to_dict


def _patched_convert_message_to_dict(
    message: BaseMessage,
    api: str = "chat/completions",
) -> dict:
    result = _original_convert_message_to_dict(message, api=api)
    if result.get("role") == "assistant" and isinstance(message, AIMessage):
        rc = message.additional_kwargs.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            result["reasoning_content"] = rc
    return result


_lc_openai_base._convert_message_to_dict = _patched_convert_message_to_dict


# ---------------------------------------------------------------------------
# System message merging (applied to ALL providers)
# ---------------------------------------------------------------------------


def _merge_system_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Merge multiple SystemMessages into a single one.

    Many providers reject requests with more than one system message.
    """
    system_parts: list[str] = []
    other_messages: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
            system_parts.append(msg.content)
        else:
            other_messages.append(msg)

    if len(system_parts) <= 1:
        return messages

    merged = SystemMessage(content="\n\n".join(system_parts))
    return [merged] + other_messages


# ---------------------------------------------------------------------------
# Provider-specific kwargs builders for init_chat_model
# ---------------------------------------------------------------------------


def _extract_options(config: ModelConfig, exclude_unset: bool = True) -> dict:
    """Extract model options as a dict."""
    if not config or not config.options:
        return {}
    if exclude_unset:
        return config.options.model_dump(exclude_unset=True)
    return config.options.model_dump()


def _build_openai_kwargs(config: ModelConfig, user_agent: str | None) -> dict:
    """Build kwargs for init_chat_model(..., model_provider='openai').

    All model options are passed through as top-level kwargs.
    ChatOpenAI natively accepts: temperature, top_p, max_tokens,
    frequency_penalty, presence_penalty, reasoning_effort, extra_body,
    and more.
    """
    options = _extract_options(config)

    kwargs: Dict[str, Any] = {**options}

    resolved_key = config.get_api_key()
    if resolved_key:
        kwargs["api_key"] = resolved_key

    resolved_url = config.get_resolved_url()
    if resolved_url:
        kwargs["base_url"] = resolved_url

    if user_agent:
        kwargs["default_headers"] = {"User-Agent": user_agent}

    return kwargs


def _build_anthropic_kwargs(config: ModelConfig, user_agent: str | None) -> dict:
    """Build kwargs for init_chat_model(..., model_provider='anthropic').

    Extracts well-known Anthropic parameters as top-level kwargs and
    passes everything else via model_kwargs.
    """
    options = _extract_options(config)

    kwargs: Dict[str, Any] = {}

    resolved_key = config.get_api_key()
    if resolved_key:
        kwargs["anthropic_api_key"] = resolved_key

    resolved_url = config.get_resolved_url()
    if resolved_url:
        kwargs["anthropic_api_url"] = resolved_url

    if user_agent:
        kwargs["default_headers"] = {"User-Agent": user_agent}

    # Well-known ChatAnthropic parameters — extract as top-level kwargs
    for key in ("temperature", "top_p", "top_k", "max_tokens", "thinking"):
        if key in options:
            kwargs[key] = options.pop(key)

    # Remaining options go through model_kwargs
    if options:
        kwargs["model_kwargs"] = options

    return kwargs


_KWARG_BUILDERS: Dict[str, Callable[[ModelConfig, str | None], dict]] = {
    "openai": _build_openai_kwargs,
    "anthropic": _build_anthropic_kwargs,
}


# ---------------------------------------------------------------------------
# Unified media extraction
# ---------------------------------------------------------------------------

# Provider-specific JSON response patterns for media extraction
_JSON_IMAGE_PATTERNS: Dict[str, list[tuple[str, str]]] = {
    "openai": [("data", "url"), ("data", "b64_json")],
}

_ADDITIONAL_KWARGS_KEYS: Dict[str, str] = {
    "openai": "image_urls",
}


def _extract_media_generic(response: BaseMessage, provider_name: str) -> List[MediaContent]:
    """Extract media from LLM response. Handles data URLs, JSON patterns,
    structured content, and additional_kwargs — parameterized by provider."""
    media_content: list[MediaContent] = []

    if not response or not hasattr(response, "content"):
        return media_content

    content = response.content

    # String content: data URLs + JSON patterns
    if isinstance(content, str):
        data_urls = extract_data_urls(content)
        for i, data_url in enumerate(data_urls):
            try:
                media = data_url_to_media_content(data_url, f"{provider_name}_generated_{i}")
                if media:
                    media_content.append(media)
            except Exception as e:
                logger.warning("Error processing data URL: %s", e)

        try:
            if content.strip().startswith(("{", "[")):
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for key, sub_key in _JSON_IMAGE_PATTERNS.get(provider_name, []):
                        items = parsed.get(key, [])
                        if not isinstance(items, list):
                            continue
                        for i, item in enumerate(items):
                            if isinstance(item, dict) and sub_key:
                                val = item.get(sub_key)
                                if val and isinstance(val, str):
                                    if sub_key == "url" and val.startswith("http"):
                                        media_content.append(
                                            MediaContent(
                                                type=MediaType.IMAGE,
                                                data=f"{provider_name} generated image URL: {val}",
                                                mime_type="image/png",
                                                filename=f"{provider_name}_image_{i}.png",
                                                source=f"{provider_name}_image_generation",
                                            )
                                        )
                                    elif sub_key == "b64_json":
                                        media_content.append(
                                            MediaContent(
                                                type=MediaType.IMAGE,
                                                data=val,
                                                mime_type="image/png",
                                                filename=f"{provider_name}_image_{i}.png",
                                                source=f"{provider_name}_image_generation",
                                            )
                                        )
                            elif isinstance(item, str) and sub_key is None:
                                media_content.append(
                                    MediaContent(
                                        type=MediaType.IMAGE,
                                        data=item,
                                        mime_type="image/png",
                                        filename=f"{provider_name}_image_{i}.png",
                                        source=f"{provider_name}_image_generation",
                                    )
                                )
        except (json.JSONDecodeError, Exception):
            pass

    # List content: structured format
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                if image_url:
                    try:
                        if image_url.startswith("data:"):
                            media = data_url_to_media_content(image_url, f"{provider_name}_structured_image")
                            if media:
                                media_content.append(media)
                        elif image_url.startswith("http"):
                            media_content.append(
                                MediaContent(
                                    type=MediaType.IMAGE,
                                    data=f"{provider_name} image URL: {image_url}",
                                    mime_type="image/png",
                                    filename=f"{provider_name}_image_from_url.png",
                                    source=f"{provider_name}_structured_content",
                                )
                            )
                    except Exception as e:
                        logger.warning("Error processing image URL: %s", e)

    # additional_kwargs
    ak_key = _ADDITIONAL_KWARGS_KEYS.get(provider_name)
    if ak_key and hasattr(response, "additional_kwargs") and isinstance(response.additional_kwargs, dict):
        items = response.additional_kwargs.get(ak_key, [])
        if isinstance(items, list):
            for i, val in enumerate(items):
                if isinstance(val, str):
                    if val.startswith("http"):
                        media_content.append(
                            MediaContent(
                                type=MediaType.IMAGE,
                                data=f"{provider_name} tool response image URL: {val}",
                                mime_type="image/png",
                                filename=f"{provider_name}_tool_image_{i}.png",
                                source=f"{provider_name}_tool_response",
                            )
                        )
                    else:
                        media_content.append(
                            MediaContent(
                                type=MediaType.IMAGE,
                                data=val,
                                mime_type="image/png",
                                filename=f"{provider_name}_tool_image_{i}.png",
                                source=f"{provider_name}_tool_response",
                            )
                        )

    return media_content


# ---------------------------------------------------------------------------
# UnifiedInferenceEngine
# ---------------------------------------------------------------------------


class UnifiedInferenceEngine(LLMInferenceEngine):
    """Single engine for all LLM providers via init_chat_model().

    Two API protocols::

        openai    — ChatOpenAI for OpenAI-compatible endpoints
        anthropic — ChatAnthropic for Anthropic-native endpoints
    """

    def __init__(
        self,
        config: ModelConfig,
        user_agent: str = None,
        agent_name: str = "CLIver",
    ):
        super().__init__(config, user_agent=user_agent, agent_name=agent_name)
        self._provider_type = config.get_provider_type()

        if self._provider_type not in _KWARG_BUILDERS:
            raise ValueError(
                f"Unknown provider type '{self._provider_type}' for model '{config.name}'. "
                f"Supported types: {', '.join(sorted(_KWARG_BUILDERS))}. "
                f"Update your config.yaml — change type to 'openai' "
                f"(for OpenAI-compatible APIs like DeepSeek, MiniMax, Ollama) "
                f"or 'anthropic' (for Anthropic-compatible APIs)."
            )

        builder = _KWARG_BUILDERS[self._provider_type]
        kwargs = builder(config, user_agent)

        self.llm = init_chat_model(
            model=config.api_model_name,
            model_provider=self._provider_type,
            **kwargs,
        )

    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        # Always merge system messages for all providers
        messages = _merge_system_messages(messages)
        return messages

    def extract_media_from_response(self, response: BaseMessage) -> List[MediaContent]:
        # Anthropic never returns inline media via the chat API
        if self._provider_type == "anthropic":
            return []
        return _extract_media_generic(response, self._provider_type)
