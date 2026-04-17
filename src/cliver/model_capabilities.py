"""
Model capabilities module for Cliver client.
Defines the capabilities of different LLM models and providers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class ProviderEnum(str, Enum):
    """Enumeration of supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"


class ModelCapability(Enum):
    """Enumeration of model capabilities."""

    TEXT_TO_TEXT = "text_to_text"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_AUDIO = "text_to_audio"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_TEXT = "image_to_text"
    AUDIO_TO_TEXT = "audio_to_text"
    VIDEO_TO_TEXT = "video_to_text"
    TOOL_CALLING = "tool_calling"
    JSON_MODE = "json_mode"
    FUNCTION_CALLING = "function_calling"
    FILE_UPLOAD = "file_upload"
    THINK_MODE = "think_mode"


# Provider-specific capability mappings
PROVIDER_CAPABILITIES = {
    "openai": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TEXT_TO_IMAGE,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.FILE_UPLOAD,
    },
    "ollama": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
    },
    "anthropic": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    "vllm": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
}

# Model-specific capability mappings (overrides provider defaults, not complement)
# The first match wins, so more specific patterns must come before shorter ones.
MODEL_CAPABILITIES = {
    # Ollama vision models
    "llava*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
    },
    "bakllava*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
    },
    "moondream*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
    },
    # Qwen models (Alibaba)
    # Qwen3-Omni: text, image, audio, video input; text + audio output
    "qwen3-omni*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.AUDIO_TO_TEXT,
        ModelCapability.VIDEO_TO_TEXT,
        ModelCapability.TEXT_TO_AUDIO,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    # Qwen VL (vision-language): text + image input
    "qwen-vl*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    "qwen3.5*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    # Qwen audio: text + audio input
    "qwen-audio*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.AUDIO_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
    # Qwen3 and later: thinking mode (hybrid reasoning) enabled by default
    "qwen3*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    # Qwen (older versions): no thinking mode
    "qwen*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
    # DeepSeek models
    # DeepSeek-R1 / deepseek-reasoner: thinking mode with reasoning_content field
    "deepseek-reasoner*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.THINK_MODE,
    },
    "deepseek-r1*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.THINK_MODE,
    },
    # DeepSeek-VL: vision-language model
    "deepseek-vl*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
    },
    # DeepSeek V3.1+: supports thinking mode (hybrid) and JSON
    "deepseek-v3*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    # DeepSeek (generic fallback, e.g. deepseek-chat)
    "deepseek*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
    # GLM models (Zhipu AI)
    # GLM-4.6V / GLM-4.5: vision + native tool calling
    "glm-4.6v*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.VIDEO_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
    "glm-4.5*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    "glm-4.7*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    # GLM-4 (generic): text + tool calling
    "glm*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
    # MiniMax models
    "minimax*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    # Claude models (Anthropic)
    "claude-opus*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    "claude-sonnet*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.THINK_MODE,
    },
    "claude-haiku*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
    "claude*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.IMAGE_TO_TEXT,
        ModelCapability.TOOL_CALLING,
        ModelCapability.JSON_MODE,
    },
    # Llama models (Meta)
    "llama3*": {
        ModelCapability.TEXT_TO_TEXT,
        ModelCapability.TOOL_CALLING,
    },
}


@dataclass
class ModelCapabilities:
    """Container for model capabilities."""

    capabilities: Set[ModelCapability]

    def supports(self, capability: ModelCapability) -> bool:
        """Check if the model supports a specific capability."""
        return capability in self.capabilities

    def supports_any(self, capabilities: Set[ModelCapability]) -> bool:
        """Check if the model supports any of the given capabilities."""
        return bool(self.capabilities.intersection(capabilities))

    def supports_all(self, capabilities: Set[ModelCapability]) -> bool:
        """Check if the model supports all of the given capabilities."""
        return self.capabilities.issuperset(capabilities)

    def get_modality_capabilities(self) -> Dict[str, bool]:
        """Get a dictionary of modality capabilities."""
        return {
            "text": ModelCapability.TEXT_TO_TEXT in self.capabilities,
            "image_input": ModelCapability.IMAGE_TO_TEXT in self.capabilities,
            "image_output": ModelCapability.TEXT_TO_IMAGE in self.capabilities,
            "audio_input": ModelCapability.AUDIO_TO_TEXT in self.capabilities,
            "audio_output": ModelCapability.TEXT_TO_AUDIO in self.capabilities,
            "video_input": ModelCapability.VIDEO_TO_TEXT in self.capabilities,
            "video_output": ModelCapability.TEXT_TO_VIDEO in self.capabilities,
            "tool_calling": ModelCapability.TOOL_CALLING in self.capabilities,
            "file_upload": ModelCapability.FILE_UPLOAD in self.capabilities,
            "think_mode": ModelCapability.THINK_MODE in self.capabilities,
        }


class ModelCapabilityDetector:
    """Detector for model capabilities based on provider and model name."""

    @staticmethod
    def detect_capabilities(provider: str, model_name: str) -> ModelCapabilities:
        """
        Detect the capabilities of a model based on its provider and name.

        Args:
            provider: The provider name (e.g., 'openai', 'ollama')
            model_name: The model name (e.g., 'qwen', 'deepseek')

        Returns:
            ModelCapabilities object with detected capabilities
        """
        # Start with provider default capabilities
        capabilities = PROVIDER_CAPABILITIES.get(provider, set())

        # Override with model-specific capabilities if available
        model_name_lower = model_name.lower()
        for pattern, model_caps in MODEL_CAPABILITIES.items():
            # Handle wildcard patterns (case-insensitive)
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if model_name_lower.startswith(prefix):
                    capabilities = model_caps
                    break
            elif model_name_lower == pattern:
                capabilities = model_caps
                break

        return ModelCapabilities(capabilities)
