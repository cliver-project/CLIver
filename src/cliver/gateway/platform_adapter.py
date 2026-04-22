"""Abstract base class for messaging platform adapters.

Provides the PlatformAdapter ABC, MessageEvent/MediaAttachment data types,
and shared utilities (message splitting) for all IM integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional


@dataclass
class MediaAttachment:
    """A media item attached to a message."""

    type: str  # "image", "file", "voice"
    data: Optional[bytes] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class MessageEvent:
    """A structured message received from a platform."""

    platform: str
    channel_id: str
    user_id: str
    text: str
    media: List[MediaAttachment] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_id: Optional[str] = None
    is_group: bool = False


# Callback type: Gateway registers this to receive messages from adapters
MessageCallback = Callable[[MessageEvent], Coroutine[Any, Any, None]]


class PlatformAdapter(ABC):
    """Base class for messaging platform integrations.

    Each adapter connects to one messaging platform (Telegram, Discord, etc.)
    and bridges messages to/from the Gateway.

    Implementors must:
    - Build MessageEvent from platform-specific updates (download media)
    - Convert markdown to platform-specific format in format_message()
    - Report max_message_length() for the platform
    - Implement all send methods (text, image, file, voice, typing)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Platform name (e.g., 'telegram', 'discord')."""

    @abstractmethod
    async def start(self, on_message: MessageCallback) -> None:
        """Connect to the platform and begin listening for messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect from the platform gracefully."""

    # --- Sending ---

    @abstractmethod
    async def send_text(self, channel_id: str, text: str, reply_to: Optional[str] = None) -> None:
        """Send a text message to a channel/user."""

    @abstractmethod
    async def send_image(self, channel_id: str, image: bytes | Path, caption: str = "") -> None:
        """Send an image with optional caption."""

    @abstractmethod
    async def send_file(self, channel_id: str, file: bytes | Path, filename: str) -> None:
        """Send a file attachment."""

    @abstractmethod
    async def send_voice(self, channel_id: str, audio: bytes | Path) -> None:
        """Send a voice message."""

    @abstractmethod
    async def send_typing(self, channel_id: str) -> None:
        """Show typing indicator."""

    # --- Formatting ---

    @abstractmethod
    def format_message(self, markdown: str) -> str:
        """Convert standard markdown to platform-specific format."""

    @abstractmethod
    def max_message_length(self) -> int:
        """Maximum message length in characters for this platform."""


def split_message(text: str, max_length: int) -> list[str]:
    """Split a long message into chunks respecting max_length.

    Tries to split at newlines first, then force-splits long lines.
    """
    if not text:
        return [""]
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = []
    current_len = 0

    for line in text.split("\n"):
        # If adding this line (+ newline) would exceed limit
        if current and current_len + len(line) + 1 > max_length:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        # If a single line exceeds max_length, force-split it
        while len(line) > max_length:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            chunks.append(line[:max_length])
            line = line[max_length:]

        current.append(line)
        current_len += len(line) + 1  # +1 for newline

    if current:
        chunks.append("\n".join(current))

    return chunks
