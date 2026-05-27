"""Abstract base class for media generation helpers (image/audio/video).

Helpers are pure request/response formatters — they build the API request
body and parse the response. The actual HTTP call is made by AgentCore,
keeping all external API calls centralized.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from cliver.media import MediaContent

logger = logging.getLogger(__name__)


class MediaGenerationHelper(ABC):
    """Base class for provider-specific media generation API formatters.

    Subclasses handle the request/response format differences between
    providers and media types (image, audio, video). They do NOT make
    HTTP calls — AgentCore handles that.
    """

    @abstractmethod
    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build the request body for the generation API.

        Args:
            prompt: Text description of the media to generate
            model_name: Model name to use for generation
            **params: Provider-specific parameters

        Returns:
            Dict to be sent as JSON body
        """
        ...

    @abstractmethod
    def parse_response(self, response_data: Dict[str, Any] | bytes) -> list[MediaContent]:
        """Parse the API response into MediaContent objects.

        Args:
            response_data: Parsed JSON (image/video) or raw bytes (audio TTS).

        Returns:
            List of MediaContent objects

        Raises:
            RuntimeError: If the API returned an error
        """
        ...


# Backward-compatible alias
ImageGenerationHelper = MediaGenerationHelper
