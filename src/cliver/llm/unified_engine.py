"""Unified LLM inference engine.

Uses LangChain's init_chat_model() for provider-agnostic ChatModel
construction. Provider-specific logic (kwargs, message conversion,
option sanitization) is handled by small dispatch tables — no
per-provider subclass needed.

Supported provider types: openai, deepseek, anthropic, ollama.
Unknown types fall back to openai-compatible behavior.
"""

import json
import logging
from typing import Any, Callable, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.media_utils import data_url_to_media_content, extract_data_urls
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System message merging (applied to ALL providers)
# ---------------------------------------------------------------------------


def _merge_system_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Merge multiple SystemMessages into a single one.

    Many providers reject requests with more than one system message.
    This collects all SystemMessage content and emits a single
    SystemMessage at the front, preserving the order of other messages.
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
# Provider-specific message converters
# ---------------------------------------------------------------------------


def _convert_deepseek_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Convert messages for DeepSeek's API.

    Flattens multipart HumanMessage content to plain strings (DeepSeek
    rejects content arrays).

    Ensures every AIMessage has reasoning_content in additional_kwargs —
    DeepSeek's API requires it on all assistant messages when thinking
    mode is active.  Messages loaded from session storage lose this
    field, so we backfill with an empty string to prevent 400 errors.
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
        elif isinstance(message, AIMessage):
            kwargs = getattr(message, "additional_kwargs", {}) or {}
            if "reasoning_content" not in kwargs:
                tool_calls = getattr(message, "tool_calls", None)
                new_kwargs = {**kwargs, "reasoning_content": ""}
                converted.append(
                    AIMessage(
                        content=message.content,
                        additional_kwargs=new_kwargs,
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    )
                )
            else:
                converted.append(message)
        else:
            converted.append(message)
    return converted


def _convert_ollama_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Convert messages for Ollama's multimedia format.

    Ollama expects images in additional_kwargs["images"] as a list of
    base64 strings, separate from the text content.
    """
    converted = []
    for message in messages:
        if isinstance(message, HumanMessage) and hasattr(message, "media_content") and message.media_content:
            image_data = []
            content_text = message.content if message.content else ""

            for media in message.media_content:
                if media.type == MediaType.IMAGE:
                    image_data.append(media.data)
                elif media.type == MediaType.AUDIO:
                    content_text += f"\n[Audio file: {media.filename}]"
                elif media.type == MediaType.VIDEO:
                    content_text += f"\n[Video file: {media.filename}]"

            converted_message = HumanMessage(content=content_text)
            if image_data:
                converted_message.additional_kwargs = {
                    **converted_message.additional_kwargs,
                    "images": image_data,
                }
            converted.append(converted_message)
        else:
            converted.append(message)
    return converted


_MESSAGE_CONVERTERS: Dict[str, Callable[[List[BaseMessage]], List[BaseMessage]]] = {
    "deepseek": _convert_deepseek_messages,
    "ollama": _convert_ollama_messages,
}


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
    options = _extract_options(config)

    kwargs: Dict[str, Any] = {
        **options,
    }

    resolved_key = config.get_api_key()
    if resolved_key:
        kwargs["api_key"] = resolved_key

    resolved_url = config.get_resolved_url()
    if resolved_url:
        kwargs["base_url"] = resolved_url

    if user_agent:
        kwargs["default_headers"] = {"User-Agent": user_agent}

    return kwargs


def _build_deepseek_kwargs(config: ModelConfig, user_agent: str | None) -> dict:
    options = _extract_options(config)

    kwargs: Dict[str, Any] = {}

    resolved_key = config.get_api_key()
    if resolved_key:
        kwargs["api_key"] = resolved_key

    resolved_url = config.get_resolved_url()
    if resolved_url:
        kwargs["api_base"] = resolved_url

    if user_agent:
        kwargs["default_headers"] = {"User-Agent": user_agent}

    for key in ("temperature", "top_p", "max_tokens"):
        if key in options:
            kwargs[key] = options[key]

    extra = {k: v for k, v in options.items() if k not in kwargs}
    if extra:
        kwargs["model_kwargs"] = extra

    return kwargs


def _build_anthropic_kwargs(config: ModelConfig, user_agent: str | None) -> dict:
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

    for key in ("temperature", "top_p", "top_k", "max_tokens"):
        if key in options:
            kwargs[key] = options[key]

    extra = {k: v for k, v in options.items() if k not in kwargs}
    if extra:
        kwargs["model_kwargs"] = extra

    return kwargs


def _build_ollama_kwargs(config: ModelConfig, user_agent: str | None) -> dict:
    options = _extract_options(config, exclude_unset=False)

    kwargs: Dict[str, Any] = {
        **options,
    }

    resolved_url = config.get_resolved_url()
    if resolved_url:
        kwargs["base_url"] = resolved_url

    if user_agent:
        kwargs["client_kwargs"] = {"headers": {"User-Agent": user_agent}}

    return kwargs


_KWARG_BUILDERS: Dict[str, Callable[[ModelConfig, str | None], dict]] = {
    "openai": _build_openai_kwargs,
    "deepseek": _build_deepseek_kwargs,
    "anthropic": _build_anthropic_kwargs,
    "ollama": _build_ollama_kwargs,
}


# ---------------------------------------------------------------------------
# Unified media extraction
# ---------------------------------------------------------------------------

# Provider-specific JSON response patterns for media extraction
_JSON_IMAGE_PATTERNS: Dict[str, list[tuple[str, str]]] = {
    "openai": [("data", "url"), ("data", "b64_json")],
    "ollama": [("images", None)],
}

_ADDITIONAL_KWARGS_KEYS: Dict[str, str] = {
    "openai": "image_urls",
    "ollama": "images",
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


# Providers that never return inline media
_NO_MEDIA_PROVIDERS = {"anthropic", "deepseek"}

_deepseek_patched = False


def _patch_deepseek_message_serialization():
    """Monkey-patch langchain's message-to-dict conversion to include
    reasoning_content and thinking content blocks for DeepSeek assistant messages.

    DeepSeek's API requires reasoning_content (or content[].thinking blocks)
    on assistant messages when thinking mode is active, but langchain's
    _convert_message_to_dict strips additional_kwargs and drops non-standard
    content blocks. This patch restores them after the original conversion.
    """
    global _deepseek_patched
    if _deepseek_patched:
        return
    _deepseek_patched = True

    try:
        import langchain_openai.chat_models.base as openai_base

        _original = openai_base._convert_message_to_dict

        def _patched(message, api="chat/completions"):
            result = _original(message, api)
            if isinstance(message, AIMessage) and result.get("role") == "assistant":
                kwargs = getattr(message, "additional_kwargs", {}) or {}
                if "reasoning_content" in kwargs:
                    result["reasoning_content"] = kwargs["reasoning_content"]
                raw_content = getattr(message, "content", None)
                if isinstance(raw_content, list):
                    thinking_blocks = [b for b in raw_content if isinstance(b, dict) and b.get("type") == "thinking"]
                    if thinking_blocks:
                        existing = result.get("content")
                        if existing is None:
                            result["content"] = thinking_blocks
                        elif isinstance(existing, list):
                            result["content"] = thinking_blocks + existing
                        else:
                            result["content"] = thinking_blocks + [{"type": "text", "text": existing}]
            return result

        openai_base._convert_message_to_dict = _patched
        logger.debug("Patched _convert_message_to_dict for DeepSeek reasoning_content")
    except Exception as e:
        logger.warning("Failed to patch DeepSeek message serialization: %s", e)


# ---------------------------------------------------------------------------
# UnifiedInferenceEngine
# ---------------------------------------------------------------------------


class UnifiedInferenceEngine(LLMInferenceEngine):
    """Single engine for all LLM providers via init_chat_model()."""

    def __init__(
        self,
        config: ModelConfig,
        user_agent: str = None,
        agent_name: str = "CLIver",
    ):
        super().__init__(config, user_agent=user_agent, agent_name=agent_name)
        self._provider_type = config.get_provider_type()
        self._effective_provider = self._detect_effective_provider(config)

        builder = _KWARG_BUILDERS.get(self._effective_provider, _build_openai_kwargs)
        kwargs = builder(config, user_agent)

        self.llm = init_chat_model(
            model=config.api_model_name,
            model_provider=self._effective_provider,
            **kwargs,
        )

        if self._effective_provider == "deepseek":
            _patch_deepseek_message_serialization()

    @staticmethod
    def _detect_effective_provider(config: ModelConfig) -> str:
        """Detect the actual API provider for message conversion.

        Some providers (e.g. DeepSeek) are configured with type "openai"
        for compatibility but need provider-specific message handling.
        Only override when the explicit type has no dedicated builder.
        """
        provider_type = config.get_provider_type()
        if provider_type not in _MESSAGE_CONVERTERS and provider_type not in _KWARG_BUILDERS:
            hints = [config.provider, config.api_model_name, config.get_resolved_url() or ""]
            for hint in hints:
                if hint and "deepseek" in hint.lower():
                    return "deepseek"
        return provider_type

    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        messages = _merge_system_messages(messages)

        converter = _MESSAGE_CONVERTERS.get(self._effective_provider)
        if converter:
            messages = converter(messages)

        return messages

    def extract_media_from_response(self, response: BaseMessage) -> List[MediaContent]:
        if self._provider_type in _NO_MEDIA_PROVIDERS:
            return []
        return _extract_media_generic(response, self._provider_type)
