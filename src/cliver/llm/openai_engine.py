import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import OpenAI

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.media_utils import (
    data_url_to_media_content,
    extract_data_urls,
)
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)


# OpenAI compatible inference engine
class OpenAICompatibleInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"):
        super().__init__(config, user_agent=user_agent, agent_name=agent_name)
        self.options = {}
        if self.config and self.config.options:
            # Only include user-specified options, not ModelOptions defaults.
            # Prevents sending unsupported params (e.g. frequency_penalty)
            # to providers that reject them (MiniMax, Qwen, etc.).
            self.options = self.config.options.model_dump(exclude_unset=True)

        # Sanitize options for provider-specific restrictions
        self._model_name_lower = (self.config.api_model_name).lower()
        self.options = _sanitize_options(self._model_name_lower, self.options)

        default_headers = {"User-Agent": user_agent} if user_agent else None

        # Resolve API key (supports vault:<service>:<key> references)
        resolved_api_key = self.config.get_api_key()
        resolved_url = self.config.get_resolved_url()

        # Initialize OpenAI client for file operations
        if resolved_api_key:
            self.openai_client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_url if resolved_url else None,
                default_headers=default_headers,
            )
        else:
            self.openai_client = None

        # Store uploaded file IDs
        self.uploaded_files = {}

        self.llm = ChatOpenAI(
            model=self.config.api_model_name,
            base_url=resolved_url,
            api_key=resolved_api_key,
            default_headers=default_headers,
            **self.options,
        )

    async def infer(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> BaseMessage:
        kwargs = _sanitize_options(self._model_name_lower, kwargs)
        return await super().infer(messages, tools, **kwargs)

    async def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        kwargs = _sanitize_options(self._model_name_lower, kwargs)
        async for chunk in super().stream(messages, tools, **kwargs):
            yield chunk

    def upload_file(self, file_path: str, purpose: str = "assistants") -> Optional[str]:
        """
        Upload a file to OpenAI for use with tools like code interpreter.

        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file (default: "assistants")

        Returns:
            File ID if successful, None if failed
        """
        if not self.openai_client:
            logger.warning("OpenAI client not initialized, cannot upload file")
            return None

        try:
            with open(file_path, "rb") as file:
                response = self.openai_client.files.create(file=file, purpose=purpose)

            file_id = response.id
            self.uploaded_files[file_path] = file_id
            logger.info(f"Uploaded file {file_path} with ID {file_id}")
            return file_id
        except Exception as e:
            # print the stack trace on exception and return None
            logger.error(f"Error uploading file {file_path}: {e}")
            return None

    def get_uploaded_file_id(self, file_path: str) -> Optional[str]:
        """
        Get the OpenAI file ID for a previously uploaded file.

        Args:
            file_path: Path to the file

        Returns:
            File ID if the file was uploaded, None otherwise
        """
        return self.uploaded_files.get(file_path)

    def list_uploaded_files(self) -> Dict[str, str]:
        """
        Get a dictionary of all uploaded files.

        Returns:
            Dictionary mapping file paths to file IDs
        """
        return self.uploaded_files.copy()

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from OpenAI.

        Args:
            file_id: ID of the file to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.openai_client:
            logger.warning("OpenAI client not initialized, cannot delete file")
            return False

        try:
            self.openai_client.files.delete(file_id)
            # Remove from our tracking
            file_path = None
            for path, id in self.uploaded_files.items():
                if id == file_id:
                    file_path = path
                    break
            if file_path:
                del self.uploaded_files[file_path]
            logger.info(f"Deleted file with ID {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False

    async def transcribe_audio(self, file_path: Path, language: Optional[str] = None) -> Optional[str]:
        if not self.openai_client:
            return None
        return await asyncio.to_thread(self._transcribe_audio_sync, file_path, language)

    def _transcribe_audio_sync(self, file_path: Path, language: Optional[str]) -> Optional[str]:
        model_name = self.config.api_model_name
        try:
            with open(file_path, "rb") as f:
                kwargs: Dict[str, Any] = {"model": model_name, "file": f}
                if language:
                    kwargs["language"] = language
                result = self.openai_client.audio.transcriptions.create(**kwargs)
            text = result.text if hasattr(result, "text") else str(result)
            if not text or not text.strip():
                return None
            logger.info("Transcribed %s via %s: %d chars", file_path.name, model_name, len(text))
            return text
        except Exception as e:
            logger.warning("Transcription failed for %s: %s", file_path.name, e)
            return None

    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Convert messages for OpenAI-compatible providers.

        Merges multiple SystemMessages into one — many providers (MiniMax,
        GLM, etc.) only accept a single system message.

        Multimodal content (images) is already in OpenAI format
        (image_url blocks) from AgentCore's message preparation.
        """
        return _merge_system_messages(messages)

    def extract_media_from_response(self, response: BaseMessage) -> List[MediaContent]:
        """
        Extract media content from OpenAI response.

        OpenAI responses may contain:
        1. Text responses from GPT models (no media content)
        2. Image URLs from DALL-E image generation
        3. Data URLs embedded in text content
        4. Base64 encoded images in DALL-E responses
        5. Special tool call responses with media

        Args:
            response: BaseMessage response from OpenAI

        Returns:
            List of MediaContent objects extracted from the response
        """
        media_content = []

        if not response or not hasattr(response, "content"):
            return media_content

        content = response.content

        # Handle string content
        if isinstance(content, str):
            # Extract data URLs from text content (if present)
            data_urls = extract_data_urls(content)
            for i, data_url in enumerate(data_urls):
                try:
                    media = data_url_to_media_content(data_url, f"openai_generated_{i}")
                    if media:
                        media_content.append(media)
                except Exception as e:
                    logger.warning(f"Error processing data URL: {e}")

            # Try to parse as JSON for structured responses (tool calls, etc.)
            try:
                if content.strip().startswith("{") or content.strip().startswith("["):
                    parsed_content = json.loads(content)

                    # Check for DALL-E image generation responses
                    # Format: {"data": [{"url": "https://..."}, {"url": "https://..."}]}
                    # Or: {"data": [{"b64_json": "base64data"}, ...]}
                    if isinstance(parsed_content, dict) and "data" in parsed_content:
                        data_items = parsed_content.get("data", [])
                        if isinstance(data_items, list):
                            for i, item in enumerate(data_items):
                                if isinstance(item, dict):
                                    # Handle URL-based images
                                    if "url" in item:
                                        url = item["url"]
                                        if url.startswith("http"):
                                            # Create a placeholder MediaContent for URL
                                            media_content.append(
                                                MediaContent(
                                                    type=MediaType.IMAGE,
                                                    data=f"OpenAI generated image URL: {url}",
                                                    mime_type="image/png",  # Default assumption
                                                    filename=f"openai_image_{i}.png",
                                                    source="openai_image_generation",
                                                )
                                            )
                                    # Handle base64 encoded images
                                    elif "b64_json" in item:
                                        b64_data = item["b64_json"]
                                        media_content.append(
                                            MediaContent(
                                                type=MediaType.IMAGE,
                                                data=b64_data,
                                                mime_type="image/png",
                                                filename=f"openai_image_{i}.png",
                                                source="openai_image_generation",
                                            )
                                        )
            except (json.JSONDecodeError, Exception):
                # Not JSON or invalid format, continue
                pass

        # Handle list content (structured format - like multimodal input messages)
        elif isinstance(content, list):
            # OpenAI's structured content format for multimodal input
            # This is typically for INPUT messages, not responses, but we check anyway
            # TODO: add handling to audio/video etc.
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            try:
                                # Handle data URLs in structured content
                                if image_url.startswith("data:"):
                                    media = data_url_to_media_content(image_url, "openai_structured_image")
                                    if media:
                                        media_content.append(media)
                                # Handle HTTP URLs
                                elif image_url.startswith("http"):
                                    # Create a placeholder MediaContent for URL
                                    media_content.append(
                                        MediaContent(
                                            type=MediaType.IMAGE,
                                            data=f"OpenAI image URL: {image_url}",
                                            mime_type="image/png",  # Default assumption
                                            filename="openai_image_from_url.png",
                                            source="openai_structured_content",
                                        )
                                    )
                            except Exception as e:
                                logger.warning(f"Error processing image URL: {e}")

        # Check for additional attributes that might contain media
        # OpenAI might put image URLs or other media in additional_kwargs
        if hasattr(response, "additional_kwargs") and isinstance(response.additional_kwargs, dict):
            additional_kwargs = response.additional_kwargs

            # Check for image URLs in tool responses or other structured data
            # TODO: shall we retrieve the image ??
            if "image_urls" in additional_kwargs:
                image_urls = additional_kwargs["image_urls"]
                if isinstance(image_urls, list):
                    for i, url in enumerate(image_urls):
                        if isinstance(url, str) and url.startswith("http"):
                            media_content.append(
                                MediaContent(
                                    type=MediaType.IMAGE,
                                    data=f"OpenAI tool response image URL: {url}",
                                    mime_type="image/png",
                                    filename=f"openai_tool_image_{i}.png",
                                    source="openai_tool_response",
                                )
                            )

        return media_content


# ---------------------------------------------------------------------------
# System message merging
# ---------------------------------------------------------------------------


def _merge_system_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Merge multiple SystemMessages into a single one.

    Many OpenAI-compatible providers (MiniMax, GLM, etc.) reject requests
    with more than one system message. This collects all SystemMessage
    content and emits a single SystemMessage at the front, preserving
    the order of all other messages.
    """
    from langchain_core.messages import SystemMessage

    system_parts: list[str] = []
    other_messages: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
            system_parts.append(msg.content)
        else:
            other_messages.append(msg)

    if len(system_parts) <= 1:
        return messages  # nothing to merge

    merged = SystemMessage(content="\n\n".join(system_parts))
    return [merged] + other_messages


# ---------------------------------------------------------------------------
# Provider-specific parameter sanitization
# ---------------------------------------------------------------------------

# Providers whose OpenAI-compatible endpoints reject frequency_penalty /
# presence_penalty (or silently ignore them, but may error in some modes).
_STRIP_PENALTY_PREFIXES = ("minimax",)

# Providers whose temperature range is (0, 1] instead of OpenAI's [0, 2].
_CLAMP_TEMP_PREFIXES = ("minimax",)

_MIN_TEMPERATURE = 0.01  # smallest value accepted by restrictive providers


def _sanitize_options(model_name: str, options: dict) -> dict:
    """Adjust inference options for provider-specific restrictions.

    Called once during engine construction. Modifies a copy, not in-place.
    """
    if not options:
        return options

    opts = dict(options)

    # Strip unsupported penalty params
    if any(model_name.startswith(p) for p in _STRIP_PENALTY_PREFIXES):
        opts.pop("frequency_penalty", None)
        opts.pop("presence_penalty", None)
        opts.pop("logit_bias", None)

    # Clamp temperature to (0, 1] for providers that reject 0 or >1
    if any(model_name.startswith(p) for p in _CLAMP_TEMP_PREFIXES):
        temp = opts.get("temperature")
        if temp is not None:
            if temp <= 0:
                opts["temperature"] = _MIN_TEMPERATURE
            elif temp > 1.0:
                opts["temperature"] = 1.0

    return opts
