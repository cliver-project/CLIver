"""Discord platform adapter using discord.py (v2.3+).

The discord.py dependency is only imported inside start() so that
the adapter class can be used/tested without installing it.
"""

import asyncio
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from cliver.config import PlatformConfig
from cliver.gateway.platform_adapter import (
    MediaAttachment,
    MessageCallback,
    MessageEvent,
    PlatformAdapter,
)

logger = logging.getLogger(__name__)


class DiscordAdapter(PlatformAdapter):
    """Discord bot adapter using discord.py."""

    def __init__(self, config: PlatformConfig):
        self._config = config
        self._token = config.token
        self._allowed_users = set(config.allowed_users) if config.allowed_users else None
        self._home_channel = config.home_channel
        self._client = None  # discord.Client
        self._on_message: Optional[MessageCallback] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "discord"

    async def start(self, on_message: MessageCallback) -> None:
        """Start the Discord bot."""
        try:
            import discord
        except ImportError as err:
            raise ImportError(
                "discord.py is required for Discord support. Install it with: pip install cliver[discord]"
            ) from err

        self._on_message = on_message

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_message(message):
            await self._handle_discord_message(message)

        # Run the client in a background task (non-blocking)
        self._task = asyncio.create_task(self._client.start(self._token))
        # Wait briefly for connection
        await asyncio.sleep(2)
        logger.info("Discord adapter started")

    async def stop(self) -> None:
        """Stop the Discord bot."""
        if self._client:
            await self._client.close()
        if self._task:
            self._task.cancel()
        logger.info("Discord adapter stopped")

    # --- Sending ---

    async def send_text(self, channel_id: str, text: str, reply_to: Optional[str] = None) -> None:
        if not self._client:
            return
        channel = self._client.get_channel(int(channel_id))
        if not channel:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            await channel.send(text)

    async def send_image(self, channel_id: str, image: bytes | Path, caption: str = "") -> None:
        if not self._client:
            return
        import discord

        channel = self._client.get_channel(int(channel_id))
        if not channel:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            if isinstance(image, bytes):
                file = discord.File(BytesIO(image), filename="image.png")
            else:
                file = discord.File(str(image))
            await channel.send(content=caption or None, file=file)

    async def send_file(self, channel_id: str, file: bytes | Path, filename: str) -> None:
        if not self._client:
            return
        import discord

        channel = self._client.get_channel(int(channel_id))
        if not channel:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            if isinstance(file, bytes):
                f = discord.File(BytesIO(file), filename=filename)
            else:
                f = discord.File(str(file), filename=filename)
            await channel.send(file=f)

    async def send_voice(self, channel_id: str, audio: bytes | Path) -> None:
        if not self._client:
            return
        import discord

        channel = self._client.get_channel(int(channel_id))
        if not channel:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            if isinstance(audio, bytes):
                f = discord.File(BytesIO(audio), filename="voice.ogg")
            else:
                f = discord.File(str(audio), filename="voice.ogg")
            await channel.send(file=f)

    async def send_typing(self, channel_id: str) -> None:
        if not self._client:
            return
        channel = self._client.get_channel(int(channel_id))
        if not channel:
            channel = await self._client.fetch_channel(int(channel_id))
        if channel:
            await channel.typing()

    # --- Formatting ---

    def format_message(self, markdown: str) -> str:
        """Discord supports standard markdown natively -- mostly pass-through."""
        return markdown

    def max_message_length(self) -> int:
        return 2000

    # --- Internal handlers ---

    def _is_allowed(self, user_id: int) -> bool:
        """Check if user is allowed (open access if no whitelist)."""
        if self._allowed_users is None:
            return True
        return str(user_id) in self._allowed_users

    async def _handle_discord_message(self, message) -> None:
        """Handle incoming Discord messages."""
        # Ignore bot's own messages
        if message.author == self._client.user:
            return
        if not self._is_allowed(message.author.id):
            return

        # Download attachments
        media = []
        for attachment in message.attachments:
            data = await attachment.read()
            if attachment.content_type and attachment.content_type.startswith("image/"):
                media_type = "image"
            elif attachment.content_type and attachment.content_type.startswith("audio/"):
                media_type = "voice"
            else:
                media_type = "file"
            media.append(
                MediaAttachment(
                    type=media_type,
                    data=data,
                    filename=attachment.filename,
                    mime_type=attachment.content_type,
                )
            )

        is_group = hasattr(message.channel, "guild") and message.channel.guild is not None

        event = MessageEvent(
            platform="discord",
            channel_id=str(message.channel.id),
            user_id=str(message.author.id),
            text=message.content or "",
            media=media,
            is_group=is_group,
        )
        if self._on_message:
            await self._on_message(event)
