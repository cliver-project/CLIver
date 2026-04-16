"""Telegram platform adapter using python-telegram-bot (v21+, async).

Provides MarkdownV2 formatting conversion and full media support.
The python-telegram-bot dependency is only imported inside start() so that
formatting functions and the adapter class can be used/tested without installing it.
"""

import logging
import re
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

# Characters that must be escaped in MarkdownV2 (outside code blocks)
_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return re.sub(r"([" + re.escape(_ESCAPE_CHARS) + r"])", r"\\\1", text)


def markdown_to_telegram(text: str) -> str:
    """Convert standard markdown to Telegram MarkdownV2.

    Preserves code blocks and inline code without escaping their contents.
    Converts **bold** to *bold*, and escapes special chars in plain text.
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
                    # Plain text — convert markdown and escape
                    converted = _convert_plain_segment(inline_seg)
                    parts.append(converted)

    return "".join(parts)


def _convert_plain_segment(text: str) -> str:
    """Convert markdown in a plain text segment and escape for MarkdownV2."""
    # Convert **bold** to *bold* (MarkdownV2 uses single *)
    text = re.sub(r"\*\*(.+?)\*\*", r"⟦BOLD⟧\1⟦/BOLD⟧", text)
    # Convert __italic__ to _italic_
    text = re.sub(r"__(.+?)__", r"⟦ITALIC⟧\1⟦/ITALIC⟧", text)
    # Escape all special chars
    text = escape_markdown_v2(text)
    # Restore bold/italic markers
    text = text.replace("⟦BOLD⟧", "*").replace("⟦/BOLD⟧", "*")
    text = text.replace("⟦ITALIC⟧", "_").replace("⟦/ITALIC⟧", "_")
    return text


class TelegramAdapter(PlatformAdapter):
    """Telegram bot adapter using python-telegram-bot."""

    def __init__(self, config: PlatformConfig):
        self._config = config
        self._token = config.token
        self._allowed_users = set(config.allowed_users) if config.allowed_users else None
        self._home_channel = config.home_channel
        self._app = None  # telegram.ext.Application
        self._on_message: Optional[MessageCallback] = None

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self, on_message: MessageCallback) -> None:
        """Start the Telegram bot with polling."""
        try:
            from telegram.ext import Application, MessageHandler, filters
        except ImportError as err:
            raise ImportError(
                "python-telegram-bot is required for Telegram support. Install it with: pip install cliver[telegram]"
            ) from err

        self._on_message = on_message
        self._app = Application.builder().token(self._token).build()

        # Handle text messages
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text_message))
        # Handle photos
        self._app.add_handler(MessageHandler(filters.PHOTO, self._on_photo_message))
        # Handle documents
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._on_document_message))
        # Handle voice
        self._app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice_message))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram adapter started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram adapter stopped")

    # --- Sending ---

    async def send_text(self, channel_id: str, text: str, reply_to: Optional[str] = None) -> None:
        if not self._app:
            return
        kwargs = {"chat_id": int(channel_id), "text": text, "parse_mode": "MarkdownV2"}
        if reply_to:
            kwargs["reply_to_message_id"] = int(reply_to)
        try:
            await self._app.bot.send_message(**kwargs)
        except Exception:
            # Fallback: send without MarkdownV2 if parsing fails
            kwargs["parse_mode"] = None
            kwargs["text"] = text.replace("\\", "")  # strip escape chars
            await self._app.bot.send_message(**kwargs)

    async def send_image(self, channel_id: str, image: bytes | Path, caption: str = "") -> None:
        if not self._app:
            return
        photo = image if isinstance(image, bytes) else open(image, "rb")
        await self._app.bot.send_photo(
            chat_id=int(channel_id),
            photo=photo,
            caption=escape_markdown_v2(caption) if caption else None,
            parse_mode="MarkdownV2" if caption else None,
        )

    async def send_file(self, channel_id: str, file: bytes | Path, filename: str) -> None:
        if not self._app:
            return
        doc = file if isinstance(file, bytes) else open(file, "rb")
        await self._app.bot.send_document(
            chat_id=int(channel_id),
            document=doc,
            filename=filename,
        )

    async def send_voice(self, channel_id: str, audio: bytes | Path) -> None:
        if not self._app:
            return
        voice = audio if isinstance(audio, bytes) else open(audio, "rb")
        await self._app.bot.send_voice(chat_id=int(channel_id), voice=voice)

    async def send_typing(self, channel_id: str) -> None:
        if not self._app:
            return
        await self._app.bot.send_chat_action(chat_id=int(channel_id), action="typing")

    # --- Formatting ---

    def format_message(self, markdown: str) -> str:
        return markdown_to_telegram(markdown)

    def max_message_length(self) -> int:
        return 4096

    # --- Internal handlers ---

    def _is_allowed(self, user_id: int) -> bool:
        """Check if user is allowed (open access if no whitelist)."""
        if self._allowed_users is None:
            return True
        return str(user_id) in self._allowed_users

    async def _on_text_message(self, update, context) -> None:
        """Handle incoming text messages."""
        if not update.effective_message or not update.effective_user:
            return
        if not self._is_allowed(update.effective_user.id):
            return

        event = MessageEvent(
            platform="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            text=update.effective_message.text or "",
            media=[],
            is_group=update.effective_chat.type in ("group", "supergroup"),
        )
        if self._on_message:
            await self._on_message(event)

    async def _on_photo_message(self, update, context) -> None:
        """Handle incoming photo messages."""
        if not update.effective_message or not update.effective_user:
            return
        if not self._is_allowed(update.effective_user.id):
            return

        # Get the largest photo
        photo = update.effective_message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        data = BytesIO()
        await file.download_to_memory(data)

        event = MessageEvent(
            platform="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            text=update.effective_message.caption or "",
            media=[MediaAttachment(type="image", data=data.getvalue(), mime_type="image/jpeg")],
            is_group=update.effective_chat.type in ("group", "supergroup"),
        )
        if self._on_message:
            await self._on_message(event)

    async def _on_document_message(self, update, context) -> None:
        """Handle incoming document messages."""
        if not update.effective_message or not update.effective_user:
            return
        if not self._is_allowed(update.effective_user.id):
            return

        doc = update.effective_message.document
        file = await context.bot.get_file(doc.file_id)
        data = BytesIO()
        await file.download_to_memory(data)

        event = MessageEvent(
            platform="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            text=update.effective_message.caption or "",
            media=[
                MediaAttachment(
                    type="file",
                    data=data.getvalue(),
                    filename=doc.file_name,
                    mime_type=doc.mime_type,
                )
            ],
            is_group=update.effective_chat.type in ("group", "supergroup"),
        )
        if self._on_message:
            await self._on_message(event)

    async def _on_voice_message(self, update, context) -> None:
        """Handle incoming voice messages."""
        if not update.effective_message or not update.effective_user:
            return
        if not self._is_allowed(update.effective_user.id):
            return

        voice = update.effective_message.voice or update.effective_message.audio
        if not voice:
            return
        file = await context.bot.get_file(voice.file_id)
        data = BytesIO()
        await file.download_to_memory(data)

        event = MessageEvent(
            platform="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            text="",
            media=[MediaAttachment(type="voice", data=data.getvalue(), mime_type="audio/ogg")],
            is_group=update.effective_chat.type in ("group", "supergroup"),
        )
        if self._on_message:
            await self._on_message(event)
