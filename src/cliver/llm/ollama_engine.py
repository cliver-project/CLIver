import json
import logging
from typing import AsyncIterator, List, Optional, Dict, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, BaseMessageChunk, AIMessageChunk
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama as Ollama

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.media_utils import (
    extract_data_urls,
    data_url_to_media_content
)
from cliver.media import MediaContent, MediaType
from cliver.model_capabilities import ModelCapability

logger = logging.getLogger(__name__)


# Ollama inference engine
class OllamaLlamaInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.options = {}
        if self.config and self.config.options:
            self.options = self.config.options.model_dump()

        self.llm = Ollama(
            model=self.config.name_in_provider or self.config.name,
            base_url=self.config.url,
            **self.options,
        )

    async def infer(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[BaseTool]],
        options: Optional[Dict[str, Any]] = None
    ) -> BaseMessage:
        try:
            # Convert messages to Ollama multimedia format if needed
            converted_messages = self._convert_messages_to_ollama_format(messages)
            _llm = await self._reconstruct_llm(self.llm, options, tools)
            response = await _llm.ainvoke(converted_messages)
            return response
        except Exception as e:
            return AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})

    async def stream(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[BaseTool]],
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[BaseMessageChunk]:
        """Stream responses from the LLM."""
        try:
            # Convert messages to Ollama multimedia format if needed
            converted_messages = self._convert_messages_to_ollama_format(messages)
            _llm = await self._reconstruct_llm(self.llm, options, tools)
            async for chunk in _llm.astream(converted_messages):
                yield chunk
        except Exception as e:
            # noinspection PyArgumentList
            yield AIMessageChunk(content=f"Error: {e}", additional_kwargs={"type": "error"})

    async def _reconstruct_llm(self, _llm: Ollama, options: dict[str, Any] | None, tools: list[BaseTool] | None) -> Ollama:
        # Create a new instance with options that override the base configuration
        if options and len(options) > 0:
            # Create base options from config if available
            ollama_options = self.options.copy()
            # Update with provided options
            ollama_options.update(options)
            _llm = Ollama(base_url=self.config.url,
                          model=self.config.name_in_provider or self.config.name,
                          **ollama_options)
        if tools and len(tools) > 0:
            # Check if the model supports tool calling
            capabilities = self.config.get_capabilities()
            if ModelCapability.TOOL_CALLING in capabilities:
                _llm = _llm.bind_tools(tools)
        return _llm

    @staticmethod
    def _convert_messages_to_ollama_format(
            messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        Convert messages to Ollama multimedia format.

        Ollama expects multimedia content in the format:
        {
            "role": "user",
            "content": "What's in this image?",
            "images": ["base64_encoded_image_data"]
        }
        """
        converted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                # Check if this is a multimedia message with custom media_content attribute
                if hasattr(message, "media_content") and message.media_content:
                    # For Ollama, we need to extract image data and put it in the images field
                    image_data = []
                    content_text = message.content if message.content else ""

                    # Extract media content
                    for media in message.media_content:
                        if media.type == MediaType.IMAGE:
                            image_data.append(media.data)
                        # For audio/video, add as text descriptions
                        # TODO: better support on audio and video
                        elif media.type == MediaType.AUDIO:
                            content_text += f"\n[Audio file: {media.filename}]"
                        elif media.type == MediaType.VIDEO:
                            content_text += f"\n[Video file: {media.filename}]"

                    # Create new message with Ollama standard format
                    converted_message = HumanMessage(content=content_text)
                    # Add image data as additional_kwargs for Ollama
                    if image_data:
                        converted_message.additional_kwargs = {
                            **converted_message.additional_kwargs,
                            "images": image_data,
                        }
                    converted_messages.append(converted_message)
                else:
                    # Not a multimedia message, keep as is
                    converted_messages.append(message)
            else:
                # Not a human message, keep as is
                converted_messages.append(message)

        return converted_messages

    def extract_media_from_response(self, response: BaseMessage) -> List[MediaContent]:
        """
        Extract media content from Ollama response.

        Ollama responses may contain:
        1. Text responses from LLM models (no media content)
        2. Data URLs embedded in text content
        3. Base64 encoded images in responses
        4. Special tool call responses with media

        Args:
            response: BaseMessage response from Ollama

        Returns:
            List of MediaContent objects extracted from the response
        """
        media_content = []

        if not response or not hasattr(response, 'content'):
            return media_content

        content = response.content

        # Handle string content
        if isinstance(content, str):
            # Extract data URLs from text content (if present)
            data_urls = extract_data_urls(content)
            for i, data_url in enumerate(data_urls):
                try:
                    media = data_url_to_media_content(data_url, f"ollama_generated_{i}")
                    if media:
                        media_content.append(media)
                except Exception as e:
                    logger.warning(f"Error processing data URL: {e}")

            # Try to parse as JSON for structured responses (tool calls, etc.)
            try:
                if content.strip().startswith('{') or content.strip().startswith('['):
                    parsed_content = json.loads(content)

                    # Check for image generation or tool responses with base64 data
                    # Ollama may return: {"images": ["base64data", ...]}
                    if isinstance(parsed_content, dict) and 'images' in parsed_content:
                        image_items = parsed_content.get('images', [])
                        if isinstance(image_items, list):
                            for i, image_data in enumerate(image_items):
                                if isinstance(image_data, str):
                                    # Handle base64 encoded images
                                    media_content.append(MediaContent(
                                        type=MediaType.IMAGE,
                                        data=image_data,
                                        mime_type="image/png",  # Default assumption
                                        filename=f"ollama_image_{i}.png",
                                        source="ollama_image_generation"
                                    ))
            except (json.JSONDecodeError, Exception):
                # Not JSON or invalid format, continue
                pass

        # Handle list content (structured format - like multimodal input messages)
        elif isinstance(content, list):
            # Ollama's structured content format for multimodal input
            # This is typically for INPUT messages, not responses, but we check anyway
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'image_url':
                        image_url = item.get('image_url', {}).get('url', '')
                        if image_url:
                            try:
                                # Handle data URLs in structured content
                                if image_url.startswith('data:'):
                                    media = data_url_to_media_content(image_url, "ollama_structured_image")
                                    if media:
                                        media_content.append(media)
                                # Handle HTTP URLs
                                elif image_url.startswith('http'):
                                    # Create a placeholder MediaContent for URL
                                    media_content.append(MediaContent(
                                        type=MediaType.IMAGE,
                                        data=f"Ollama image URL: {image_url}",
                                        mime_type="image/png",  # Default assumption
                                        filename="ollama_image_from_url.png",
                                        source="ollama_structured_content"
                                    ))
                            except Exception as e:
                                logger.warning(f"Error processing image URL: {e}")

        # Check for additional attributes that might contain media
        # Ollama might put image data or other media in additional_kwargs
        if hasattr(response, 'additional_kwargs') and isinstance(response.additional_kwargs, dict):
            additional_kwargs = response.additional_kwargs

            # Check for images in additional_kwargs (similar to input format)
            if 'images' in additional_kwargs:
                images = additional_kwargs['images']
                if isinstance(images, list):
                    for i, image_data in enumerate(images):
                        if isinstance(image_data, str):
                            # Handle base64 encoded images
                            media_content.append(MediaContent(
                                type=MediaType.IMAGE,
                                data=image_data,
                                mime_type="image/png",
                                filename=f"ollama_tool_image_{i}.png",
                                source="ollama_tool_response"
                            ))

        return media_content

