"""Abstract base class for image generation helpers.

Helpers are pure request/response formatters — they build the API request
body and parse the response. The actual HTTP call is made by AgentCore,
keeping all external API calls centralized.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from cliver.media import MediaContent

logger = logging.getLogger(__name__)


class ImageGenerationHelper(ABC):
    """Base class for provider-specific image generation API formatters.

    Subclasses handle the request/response format differences between
    providers (MiniMax, OpenAI, DashScope, etc.). They do NOT make
    HTTP calls — AgentCore handles that.
    """

    @abstractmethod
    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build the request body for the image generation API.

        Args:
            prompt: Text description of the image
            model_name: Model name override (uses adapter default if None)
            **params: Provider-specific parameters

        Returns:
            Dict to be sent as JSON body
        """
        ...

    @abstractmethod
    def parse_response(self, response_data: Dict[str, Any]) -> list[MediaContent]:
        """Parse the API response into MediaContent objects.

        Args:
            response_data: Parsed JSON response from the API

        Returns:
            List of MediaContent with type=IMAGE

        Raises:
            RuntimeError: If the API returned an error
        """
        ...
