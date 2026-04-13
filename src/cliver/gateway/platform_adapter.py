"""Abstract base class for messaging platform adapters."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

# Type for the message callback: async fn(platform_name, channel_id, user_id, text) -> response_text
MessageCallback = Callable[[str, str, str, str], Coroutine[Any, Any, str]]


class PlatformAdapter(ABC):
    """Base class for messaging platform integrations.

    Each adapter connects to one messaging platform (Telegram, Discord, etc.)
    and bridges messages to/from the gateway's TaskExecutor.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Platform name (e.g., 'telegram', 'discord')."""

    @abstractmethod
    async def start(self, on_message: MessageCallback) -> None:
        """Connect to the platform and begin listening for messages.

        Args:
            on_message: Callback to invoke when a message is received.
                        Returns the response text to send back.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect from the platform gracefully."""

    @abstractmethod
    async def send_response(self, channel_id: str, text: str) -> None:
        """Send a response message to a specific channel/user.

        Args:
            channel_id: Platform-specific channel or user identifier.
            text: The response text to send.
        """
