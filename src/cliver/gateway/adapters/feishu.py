"""Feishu (飞书/Lark) platform adapter.

Uses Feishu Open API with event subscriptions for receiving messages
and REST API for sending. The aiohttp dependency is only imported inside
start() so that the adapter class can be used/tested without installing it.
"""

import logging
import time
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

# Feishu Open API base URL
FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuAdapter(PlatformAdapter):
    """Feishu (飞书/Lark) adapter using webhook callbacks and REST API."""

    def __init__(self, config: PlatformConfig):
        self._config = config
        self._token = config.token  # app secret
        self._app_id = getattr(config, "app_id", None)
        self._verification_token = getattr(config, "verification_token", None)
        self._allowed_users = set(config.allowed_users) if config.allowed_users else None
        self._home_channel = config.home_channel
        self._on_message: Optional[MessageCallback] = None
        self._tenant_access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._session = None  # aiohttp.ClientSession

    @property
    def name(self) -> str:
        return "feishu"

    async def start(self, on_message: MessageCallback) -> None:
        """Start the Feishu adapter in webhook mode.

        Messages arrive via HTTP event subscriptions through the Gateway's API server.
        This method initializes the HTTP client and fetches a tenant access token.
        """
        try:
            import aiohttp
        except ImportError as err:
            raise ImportError(
                "aiohttp is required for Feishu support. Install it with: pip install cliver[feishu]"
            ) from err

        self._on_message = on_message
        self._session = aiohttp.ClientSession()

        # Fetch initial tenant access token
        if self._app_id and self._token:
            await self._refresh_tenant_access_token()

        logger.info("Feishu adapter started (webhook mode)")

    async def stop(self) -> None:
        """Stop the Feishu adapter."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Feishu adapter stopped")

    # --- Sending ---

    async def send_text(self, channel_id: str, text: str, reply_to: Optional[str] = None) -> None:
        token = await self._get_tenant_access_token()
        if not token or not self._session:
            return
        url = f"{FEISHU_API_BASE}/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        params = {"receive_id_type": "chat_id"}
        payload = {
            "receive_id": channel_id,
            "msg_type": "text",
            "content": f'{{"text": "{text}"}}',
        }
        if reply_to:
            payload["reply_in_thread"] = True
        async with self._session.post(url, headers=headers, params=params, json=payload) as resp:
            if resp.status != 200:
                logger.warning("Feishu send_text failed: %s", resp.status)

    async def send_image(self, channel_id: str, image: bytes | Path, caption: str = "") -> None:
        token = await self._get_tenant_access_token()
        if not token or not self._session:
            return
        # Upload image first
        image_key = await self._upload_image(image)
        if not image_key:
            return
        url = f"{FEISHU_API_BASE}/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        params = {"receive_id_type": "chat_id"}
        payload = {
            "receive_id": channel_id,
            "msg_type": "image",
            "content": f'{{"image_key": "{image_key}"}}',
        }
        async with self._session.post(url, headers=headers, params=params, json=payload) as resp:
            if resp.status != 200:
                logger.warning("Feishu send_image failed: %s", resp.status)

        # Send caption as a separate text message if provided
        if caption:
            await self.send_text(channel_id, caption)

    async def send_file(self, channel_id: str, file: bytes | Path, filename: str) -> None:
        token = await self._get_tenant_access_token()
        if not token or not self._session:
            return
        # Upload file first
        file_key = await self._upload_file(file, filename)
        if not file_key:
            return
        url = f"{FEISHU_API_BASE}/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        params = {"receive_id_type": "chat_id"}
        payload = {
            "receive_id": channel_id,
            "msg_type": "file",
            "content": f'{{"file_key": "{file_key}"}}',
        }
        async with self._session.post(url, headers=headers, params=params, json=payload) as resp:
            if resp.status != 200:
                logger.warning("Feishu send_file failed: %s", resp.status)

    async def send_voice(self, channel_id: str, audio: bytes | Path) -> None:
        token = await self._get_tenant_access_token()
        if not token or not self._session:
            return
        # Upload audio as a file
        file_key = await self._upload_file(audio, "voice.opus")
        if not file_key:
            return
        url = f"{FEISHU_API_BASE}/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        params = {"receive_id_type": "chat_id"}
        payload = {
            "receive_id": channel_id,
            "msg_type": "audio",
            "content": f'{{"file_key": "{file_key}"}}',
        }
        async with self._session.post(url, headers=headers, params=params, json=payload) as resp:
            if resp.status != 200:
                logger.warning("Feishu send_voice failed: %s", resp.status)

    async def send_typing(self, channel_id: str) -> None:
        # Feishu does not support typing indicators via API
        pass

    # --- Formatting ---

    def format_message(self, markdown: str) -> str:
        """Feishu supports markdown natively in certain message types -- pass through."""
        return markdown

    def max_message_length(self) -> int:
        return 4000

    # --- Internal helpers ---

    def _is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed (open access if no whitelist)."""
        if self._allowed_users is None:
            return True
        return str(user_id) in self._allowed_users

    async def _refresh_tenant_access_token(self) -> None:
        """Fetch a new tenant access token from the Feishu API."""
        if not self._session or not self._app_id or not self._token:
            return
        url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self._app_id,
            "app_secret": self._token,
        }
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("code") == 0:
                        self._tenant_access_token = data["tenant_access_token"]
                        self._token_expires_at = time.time() + data.get("expire", 7200) - 300
                        logger.debug("Feishu tenant access token refreshed")
                    else:
                        logger.warning("Feishu token error: %s", data.get("msg"))
        except Exception as e:
            logger.warning("Failed to refresh Feishu tenant access token: %s", e)

    async def _get_tenant_access_token(self) -> Optional[str]:
        """Get a valid tenant access token, refreshing if needed."""
        if not self._tenant_access_token or time.time() >= self._token_expires_at:
            await self._refresh_tenant_access_token()
        return self._tenant_access_token

    async def _upload_image(self, image: bytes | Path) -> Optional[str]:
        """Upload an image to Feishu and return image_key."""
        token = await self._get_tenant_access_token()
        if not token or not self._session:
            return None

        import aiohttp

        url = f"{FEISHU_API_BASE}/im/v1/images"
        headers = {"Authorization": f"Bearer {token}"}
        data = aiohttp.FormData()
        data.add_field("image_type", "message")
        if isinstance(image, Path):
            with open(image, "rb") as f:
                data.add_field("image", f, filename="image.png")
                async with self._session.post(url, headers=headers, data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("data", {}).get("image_key")
        else:
            data.add_field("image", BytesIO(image), filename="image.png")
            async with self._session.post(url, headers=headers, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("data", {}).get("image_key")

        return None

    async def _upload_file(self, file: bytes | Path, filename: str) -> Optional[str]:
        """Upload a file to Feishu and return file_key."""
        token = await self._get_tenant_access_token()
        if not token or not self._session:
            return None

        import aiohttp

        url = f"{FEISHU_API_BASE}/im/v1/files"
        headers = {"Authorization": f"Bearer {token}"}
        data = aiohttp.FormData()
        data.add_field("file_type", "stream")
        data.add_field("file_name", filename)
        if isinstance(file, Path):
            with open(file, "rb") as f:
                data.add_field("file", f, filename=filename)
                async with self._session.post(url, headers=headers, data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("data", {}).get("file_key")
        else:
            data.add_field("file", BytesIO(file), filename=filename)
            async with self._session.post(url, headers=headers, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("data", {}).get("file_key")

        return None

    async def handle_webhook(self, request_data: dict) -> Optional[dict]:
        """Process an incoming webhook event from Feishu.

        This method is called by the Gateway's API server when an event
        arrives at the /webhook/feishu endpoint.

        Returns a challenge response dict for URL verification, or None.
        """
        # Handle URL verification challenge
        if "challenge" in request_data:
            return {"challenge": request_data["challenge"]}

        # Handle event callback
        header = request_data.get("header", {})
        event = request_data.get("event", {})

        # Verify token if configured
        if self._verification_token:
            token = header.get("token", "")
            if token != self._verification_token:
                logger.warning("Feishu webhook verification failed")
                return None

        event_type = header.get("event_type", "")
        if event_type != "im.message.receive_v1":
            return None

        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {}).get("open_id", "")
        if not sender_id or not self._is_allowed(sender_id):
            return None

        message = event.get("message", {})
        msg_type = message.get("message_type", "")
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")

        text = ""
        media = []

        if msg_type == "text":
            import json

            try:
                content = json.loads(message.get("content", "{}"))
                text = content.get("text", "")
            except (json.JSONDecodeError, TypeError):
                text = ""
        elif msg_type == "image":
            import json

            try:
                content = json.loads(message.get("content", "{}"))
                image_key = content.get("image_key", "")
                if image_key:
                    media.append(MediaAttachment(type="image", url=image_key))
            except (json.JSONDecodeError, TypeError):
                pass
        elif msg_type == "file":
            import json

            try:
                content = json.loads(message.get("content", "{}"))
                file_key = content.get("file_key", "")
                file_name = content.get("file_name", "file")
                if file_key:
                    media.append(MediaAttachment(type="file", url=file_key, filename=file_name))
            except (json.JSONDecodeError, TypeError):
                pass
        elif msg_type == "audio":
            import json

            try:
                content = json.loads(message.get("content", "{}"))
                file_key = content.get("file_key", "")
                if file_key:
                    media.append(MediaAttachment(type="voice", url=file_key))
            except (json.JSONDecodeError, TypeError):
                pass

        msg_event = MessageEvent(
            platform="feishu",
            channel_id=chat_id,
            user_id=sender_id,
            text=text,
            media=media,
            reply_to_message_id=message.get("parent_id"),
            is_group=chat_type == "group",
        )
        if self._on_message:
            await self._on_message(msg_event)

        return None
