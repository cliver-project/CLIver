import json
import logging
import os
from typing import AsyncIterator, List, Optional, Dict, Any
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import OpenAI

from cliver.config import ModelConfig
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.media_utils import (
    extract_data_urls,
    data_url_to_media_content,
    extract_media_from_json,
    get_file_extension
)
from cliver.media import MediaContent, MediaType
from cliver.model_capabilities import ModelCapability

logger = logging.getLogger(__name__)


# OpenAI compatible inference engine
class OpenAICompatibleInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Initialize OpenAI client for file operations
        if self.config.api_key:
            self.openai_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.url if self.config.url else None
            )
        else:
            self.openai_client = None

        # Store uploaded file IDs
        self.uploaded_files = {}

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
                response = self.openai_client.files.create(
                    file=file,
                    purpose=purpose
                )

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

    async def infer(
        self, messages: list[BaseMessage], tools: Optional[list[BaseTool]]
    ) -> BaseMessage:
        try:
            # Convert messages to OpenAI multi-media format if needed
            converted_messages = self._convert_messages_to_openai_format(
                messages)
            _llm = self.llm
            if tools:
                # Check if the model supports tool calling
                capabilities = self.config.get_capabilities()
                if ModelCapability.TOOL_CALLING in capabilities:
                    _llm = self.llm.bind_tools(tools, strict=True)
            response = await _llm.ainvoke(converted_messages)
            return response
        except Exception as e:
            return AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})

    async def stream(
        self, messages: list[BaseMessage], tools: Optional[list[BaseTool]]
    ) -> AsyncIterator[BaseMessage]:
        """Stream responses from the LLM."""
        # Convert messages to OpenAI multi-media format if needed
        converted_messages = self._convert_messages_to_openai_format(messages)
        _llm = self.llm
        if tools:
            # Check if the model supports tool calling
            capabilities = self.config.get_capabilities()
            if ModelCapability.TOOL_CALLING in capabilities:
                _llm = self.llm.bind_tools(tools, strict=True)
        try:
            async for chunk in _llm.astream(converted_messages):
                yield chunk
        except Exception as e:
            yield AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})

    @staticmethod
    def _convert_messages_to_openai_format(
            messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        Convert messages to OpenAI multimedia format.

        OpenAI expects multimedia content in the format:
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/..."
                    }
                }
            ]
        }
        """
        converted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                # Check if this is a multimedia message with custom media_content attribute
                if hasattr(message, "media_content") and message.media_content:
                    # Convert from custom format to OpenAI standard format
                    content_parts = []

                    # Add text content if present
                    if message.content:
                        content_parts.append(
                            {"type": "text", "text": message.content})

                    # Add media content using shared utility function
                    from cliver.media import add_media_content_to_message_parts
                    add_media_content_to_message_parts(content_parts, message.media_content)

                    # Create new message with OpenAI standard format
                    converted_message = HumanMessage(content=content_parts)
                    converted_messages.append(converted_message)
                else:
                    # Not a multi-media message, keep as is
                    converted_messages.append(message)
            else:
                # Not a human message, keep as is
                converted_messages.append(message)

        return converted_messages

    def system_message(self) -> str:
        """
        System message optimized for OpenAI-compatible models to ensure stable JSON tool calling format.
        """
        return """
You are an AI assistant that can use tools to help answer questions.

Available tools will be provided to you. When you need to use a tool, you MUST use the exact tool name provided.

To use a tool, respond ONLY with a JSON object in this exact format:
{
  "tool_calls": [
    {
      "name": "exact_tool_name",
      "args": {
        "argument_name": "argument_value"
      },
      "id": "unique_identifier_for_this_call",
      "type": "tool_call"
    }
  ]
}

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. Only use the exact tool names provided to you
2. Respond ONLY with the JSON format when calling tools - no other text or explanation
3. Generate a unique ID for each tool call using standard UUID format
4. Always include the "type": "tool_call" field
5. Ensure your JSON is properly formatted and parsable
6. The tool_calls array must be a valid JSON array
7. Do not embed tool calls in markdown code blocks or any other formatting

After you make a tool call, you will receive the result. You may need to make additional tool calls based on the results until you have enough information to provide your final answer. The process can involve multiple rounds of tool calls.

If you have all the information needed to answer directly without using any tools, provide a text response.

Examples of CORRECT tool usage:
{"tool_calls": [{"name": "get_current_weather", "args": {"location": "New York"}, "id": "call_1234567890abcdef", "type": "tool_call"}]}

Examples of INCORRECT tool usage:
- Using markdown code blocks
- Adding explanatory text before/after JSON
- Missing required fields
- Improperly formatted JSON
"""

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

        if not response or not hasattr(response, 'content'):
            return media_content

        content = response.content

        # Handle string content
        if isinstance(content, str):
            # Extract data URLs from text content (if present)
            data_urls = extract_data_urls(content)
            for i, data_url in enumerate(data_urls):
                try:
                    media = data_url_to_media_content(
                        data_url, f"openai_generated_{i}")
                    if media:
                        media_content.append(media)
                except Exception as e:
                    logger.warning(f"Error processing data URL: {e}")

            # Try to parse as JSON for structured responses (tool calls, etc.)
            try:
                if content.strip().startswith('{') or content.strip().startswith('['):
                    parsed_content = json.loads(content)

                    # Check for DALL-E image generation responses
                    # Format: {"data": [{"url": "https://..."}, {"url": "https://..."}]}
                    # Or: {"data": [{"b64_json": "base64data"}, ...]}
                    if isinstance(parsed_content, dict) and 'data' in parsed_content:
                        data_items = parsed_content.get('data', [])
                        if isinstance(data_items, list):
                            for i, item in enumerate(data_items):
                                if isinstance(item, dict):
                                    # Handle URL-based images
                                    if 'url' in item:
                                        url = item['url']
                                        if url.startswith('http'):
                                            # Create a placeholder MediaContent for URL
                                            media_content.append(MediaContent(
                                                type=MediaType.IMAGE,
                                                data=f"OpenAI generated image URL: {url}",
                                                mime_type="image/png",  # Default assumption
                                                filename=f"openai_image_{i}.png",
                                                source="openai_image_generation"
                                            ))
                                    # Handle base64 encoded images
                                    elif 'b64_json' in item:
                                        b64_data = item['b64_json']
                                        media_content.append(MediaContent(
                                            type=MediaType.IMAGE,
                                            data=b64_data,
                                            mime_type="image/png",
                                            filename=f"openai_image_{i}.png",
                                            source="openai_image_generation"
                                        ))
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
                    if item.get('type') == 'image_url':
                        image_url = item.get('image_url', {}).get('url', '')
                        if image_url:
                            try:
                                # Handle data URLs in structured content
                                if image_url.startswith('data:'):
                                    media = data_url_to_media_content(
                                        image_url, "openai_structured_image")
                                    if media:
                                        media_content.append(media)
                                # Handle HTTP URLs
                                elif image_url.startswith('http'):
                                    # Create a placeholder MediaContent for URL
                                    media_content.append(MediaContent(
                                        type=MediaType.IMAGE,
                                        data=f"OpenAI image URL: {image_url}",
                                        mime_type="image/png",  # Default assumption
                                        filename="openai_image_from_url.png",
                                        source="openai_structured_content"
                                    ))
                            except Exception as e:
                                logger.warning(
                                    f"Error processing image URL: {e}")

        # Check for additional attributes that might contain media
        # OpenAI might put image URLs or other media in additional_kwargs
        if hasattr(response, 'additional_kwargs') and isinstance(response.additional_kwargs, dict):
            additional_kwargs = response.additional_kwargs

            # Check for image URLs in tool responses or other structured data
            # TODO: shall we retrieve the image ??
            if 'image_urls' in additional_kwargs:
                image_urls = additional_kwargs['image_urls']
                if isinstance(image_urls, list):
                    for i, url in enumerate(image_urls):
                        if isinstance(url, str) and url.startswith('http'):
                            media_content.append(MediaContent(
                                type=MediaType.IMAGE,
                                data=f"OpenAI tool response image URL: {url}",
                                mime_type="image/png",
                                filename=f"openai_tool_image_{i}.png",
                                source="openai_tool_response"
                            ))

        return media_content
