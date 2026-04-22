"""Slack platform adapter using slack-bolt (async, Socket Mode).

Provides mrkdwn formatting conversion and full media support.
The slack-bolt and slack-sdk dependencies are only imported inside start() so that
the adapter class can be used/tested without installing them.
"""

import logging
import re
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


def markdown_to_mrkdwn(text: str) -> str:
    """Convert standard markdown to Slack mrkdwn format.

    Slack uses *bold* (not **bold**), _italic_ (same), ~strikethrough~,
    and preserves `code` and ```code blocks```.
    """
    parts = []
    # Split by code blocks first (```...```)
    code_block_pattern = re.compile(r"(```[\s\S]*?```)", re.DOTALL)
    segments = code_block_pattern.split(text)

    for segment in segments:
        if segment.startswith("```") and segment.endswith("```"):
            # Code block — preserve as-is
            parts.append(segment)
        else:
            # Split by inline code (`...`)
            inline_pattern = re.compile(r"(`[^`]+`)")
            inline_segments = inline_pattern.split(segment)
            for inline_seg in inline_segments:
                if inline_seg.startswith("`") and inline_seg.endswith("`"):
                    # Inline code — preserve as-is
                    parts.append(inline_seg)
                else:
                    # Convert **bold** to *bold*
                    converted = re.sub(r"\*\*(.+?)\*\*", r"*\1*", inline_seg)
                    parts.append(converted)

    return "".join(parts)


class SlackAdapter(PlatformAdapter):
    """Slack bot adapter using slack-bolt with Socket Mode."""

    def __init__(self, config: PlatformConfig):
        self._config = config
        self._bot_token = config.token
        self._app_token = getattr(config, "app_token", None)
        self._allowed_users = set(config.allowed_users) if config.allowed_users else None
        self._home_channel = config.home_channel
        self._app = None  # AsyncApp
        self._handler = None  # AsyncSocketModeHandler
        self._client = None  # AsyncWebClient
        self._on_message: Optional[MessageCallback] = None

    @property
    def name(self) -> str:
        return "slack"

    async def start(self, on_message: MessageCallback) -> None:
        """Start the Slack bot with Socket Mode."""
        try:
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
            from slack_bolt.async_app import AsyncApp
        except ImportError as err:
            raise ImportError(
                "slack-bolt is required for Slack support. Install it with: pip install cliver[slack]"
            ) from err

        if not self._app_token:
            raise ValueError("Slack Socket Mode requires an 'app_token' in platform config")

        self._on_message = on_message
        self._app = AsyncApp(token=self._bot_token)
        self._client = self._app.client

        # Verify bot token before starting socket mode
        try:
            auth = await self._client.auth_test()
            if not auth.get("ok"):
                raise ConnectionError(f"Slack auth failed: {auth.get('error', 'unknown')}")
            logger.info(f"Slack authenticated as {auth.get('bot_id', '?')}")
        except Exception as e:
            if "auth" in str(e).lower() or "invalid" in str(e).lower():
                raise ConnectionError(f"Slack bot token invalid: {e}") from e
            raise

        @self._app.event("message")
        async def handle_message_event(event, say):
            logger.info("Slack event 'message' received: %s", event)
            await self._on_slack_message(event, say, self._client)

        @self._app.middleware
        async def log_all_events(body, next):
            logger.info(
                "Slack middleware: type=%s, event_type=%s",
                body.get("type", "?"),
                body.get("event", {}).get("type", "?") if isinstance(body.get("event"), dict) else "?",
            )
            await next()

        self._handler = AsyncSocketModeHandler(self._app, self._app_token)
        await self._handler.connect_async()
        logger.info("Slack adapter started (Socket Mode)")

    async def stop(self) -> None:
        """Stop the Slack bot."""
        if self._handler:
            await self._handler.close_async()
        logger.info("Slack adapter stopped")

    # --- Sending ---

    async def send_text(self, channel_id: str, text: str, reply_to: Optional[str] = None) -> None:
        if not self._client:
            return
        kwargs = {"channel": channel_id, "text": text}
        if reply_to:
            kwargs["thread_ts"] = reply_to
        await self._client.chat_postMessage(**kwargs)

    async def send_image(self, channel_id: str, image: bytes | Path, caption: str = "") -> None:
        if not self._client:
            return
        if isinstance(image, Path):
            await self._client.files_upload_v2(
                channel=channel_id,
                file=str(image),
                initial_comment=caption or None,
            )
        else:
            await self._client.files_upload_v2(
                channel=channel_id,
                content=image,
                filename="image.png",
                initial_comment=caption or None,
            )

    async def send_file(self, channel_id: str, file: bytes | Path, filename: str) -> None:
        if not self._client:
            return
        if isinstance(file, Path):
            await self._client.files_upload_v2(
                channel=channel_id,
                file=str(file),
                filename=filename,
            )
        else:
            await self._client.files_upload_v2(
                channel=channel_id,
                content=file,
                filename=filename,
            )

    async def send_voice(self, channel_id: str, audio: bytes | Path) -> None:
        if not self._client:
            return
        if isinstance(audio, Path):
            await self._client.files_upload_v2(
                channel=channel_id,
                file=str(audio),
                filename="voice.ogg",
            )
        else:
            await self._client.files_upload_v2(
                channel=channel_id,
                content=audio,
                filename="voice.ogg",
            )

    async def send_typing(self, channel_id: str) -> None:
        # Slack doesn't have a direct typing indicator API for bots.
        # This is a no-op for Slack.
        pass

    # --- Formatting ---

    def format_message(self, markdown: str) -> str:
        return markdown_to_mrkdwn(markdown)

    def max_message_length(self) -> int:
        return 4000

    # --- Internal handlers ---

    def _is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed (open access if no whitelist)."""
        if self._allowed_users is None:
            return True
        return str(user_id) in self._allowed_users

    async def _on_slack_message(self, message: dict, say, client) -> None:
        """Handle incoming Slack messages."""
        logger.info(
            "Slack message received: user=%s, channel=%s, text=%.100s",
            message.get("user", "?"),
            message.get("channel", "?"),
            message.get("text", ""),
        )
        user_id = message.get("user", "")
        if not user_id:
            logger.debug("Slack message ignored: no user_id")
            return
        if not self._is_allowed(user_id):
            logger.info("Slack message ignored: user %s not in allowed list %s", user_id, self._allowed_users)
            return

        # Determine if it's a group (channel) or DM
        channel_type = message.get("channel_type", "")
        is_group = channel_type in ("channel", "group")

        text = message.get("text", "")
        media = []

        # Handle file attachments
        files = message.get("files", [])
        for file_info in files:
            file_url = file_info.get("url_private")
            if not file_url:
                continue

            # Download the file using the bot token for auth
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self._bot_token}"}
                    async with session.get(file_url, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                        else:
                            logger.warning("Failed to download Slack file: %s", resp.status)
                            continue
            except ImportError:
                logger.warning("aiohttp required to download Slack file attachments")
                continue

            mimetype = file_info.get("mimetype", "")
            if mimetype.startswith("image/"):
                media_type = "image"
            elif mimetype.startswith("audio/"):
                media_type = "voice"
            else:
                media_type = "file"

            media.append(
                MediaAttachment(
                    type=media_type,
                    data=data,
                    filename=file_info.get("name"),
                    mime_type=mimetype,
                )
            )

        event = MessageEvent(
            platform="slack",
            channel_id=message.get("channel", ""),
            user_id=user_id,
            text=text,
            media=media,
            reply_to_message_id=message.get("thread_ts"),
            is_group=is_group,
        )
        if self._on_message:
            await self._on_message(event)
