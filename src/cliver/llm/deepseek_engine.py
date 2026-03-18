"""DeepSeek-specific inference engine.

Handles DeepSeek API quirks that differ from standard OpenAI-compatible APIs:
- reasoning_content field in responses (for DeepSeek-R1 / deepseek-reasoner)
- Content array flattening (DeepSeek requires content as string, not array)
- reasoning_content MUST be preserved in assistant messages in conversation history

Langchain's ChatOpenAI drops reasoning_content in both directions:
- Response parsing (_convert_dict_to_message): only captures known OpenAI fields
- Request serialization (_convert_message_to_dict): only serializes known OpenAI fields

ChatDeepSeek fixes both by overriding _create_chat_result and _get_request_payload.
"""

import logging
from typing import Any, List, Optional, Union

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from cliver.config import ModelConfig
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine

logger = logging.getLogger(__name__)


class ChatDeepSeek(ChatOpenAI):
    """ChatOpenAI subclass that preserves reasoning_content for DeepSeek's API.

    Langchain drops reasoning_content in both directions because it's not a
    standard OpenAI field. DeepSeek's thinking mode API requires it on all
    assistant messages in conversation history.
    """

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        """Capture reasoning_content from DeepSeek's response into additional_kwargs."""
        result = super()._create_chat_result(response, generation_info=generation_info)

        # Extract reasoning_content from the raw response
        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices", [])

        for i, gen in enumerate(result.generations):
            if i < len(choices):
                reasoning = choices[i].get("message", {}).get("reasoning_content")
                if reasoning is not None and isinstance(gen.message, AIMessage):
                    gen.message.additional_kwargs["reasoning_content"] = reasoning

        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Re-inject reasoning_content into assistant messages in the API payload."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Collect reasoning_content from the original AIMessages
        messages = self._convert_input(input_).to_messages()
        ai_reasoning: dict[int, str] = {}
        ai_index = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                rc = (getattr(msg, "additional_kwargs", None) or {}).get("reasoning_content")
                if rc is not None:
                    ai_reasoning[ai_index] = rc
                ai_index += 1

        # Re-inject reasoning_content into serialized assistant message dicts
        if ai_reasoning and "messages" in payload:
            ai_index = 0
            for msg_dict in payload["messages"]:
                if msg_dict.get("role") == "assistant":
                    if ai_index in ai_reasoning:
                        msg_dict["reasoning_content"] = ai_reasoning[ai_index]
                    ai_index += 1

        return payload


class DeepSeekInferenceEngine(OpenAICompatibleInferenceEngine):
    def __init__(self, config: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"):
        super().__init__(config, user_agent=user_agent, agent_name=agent_name)

        # Replace the default ChatOpenAI with ChatDeepSeek to preserve reasoning_content
        default_headers = {"User-Agent": user_agent} if user_agent else None
        resolved_api_key = self.config.get_api_key()
        self.llm = ChatDeepSeek(
            model=self.config.name_in_provider or self.config.name,
            base_url=self.config.url,
            api_key=resolved_api_key,
            default_headers=default_headers,
            **self.options,
        )

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
