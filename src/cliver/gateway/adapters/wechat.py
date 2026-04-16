"""WeChat (WeCom/企业微信) platform adapter.

Uses WeCom webhook API for receiving messages and REST API for sending.
The aiohttp dependency is only imported inside start() so that
the adapter class can be used/tested without installing it.
"""

import logging
import re
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

# WeCom API base URL
WECOM_API_BASE = "https://qyapi.weixin.qq.com/cgi-bin"


def strip_markdown(text: str) -> str:
    """Convert standard markdown to plain text for WeChat.

    WeChat text messages have limited markdown support,
    so we strip formatting to plain text.
    """
    # Remove code blocks — keep the code content
    text = re.sub(r"```(?:\w+)?\n?([\s\S]*?)```", r"\1", text)
    # Remove bold markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Remove italic markers (underscore)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Remove italic markers (asterisk)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Remove inline code markers
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Remove link markdown, keep text
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text


class WeChatAdapter(PlatformAdapter):
    """WeChat (WeCom) adapter using webhook callbacks and REST API."""

    def __init__(self, config: PlatformConfig):
        self._config = config
        self._token = config.token  # corp secret
        self._corp_id = getattr(config, "corp_id", None)
        self._agent_id = getattr(config, "agent_id", None)
        self._allowed_users = set(config.allowed_users) if config.allowed_users else None
        self._home_channel = config.home_channel
        self._on_message: Optional[MessageCallback] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._session = None  # aiohttp.ClientSession

    @property
    def name(self) -> str:
        return "wechat"

    async def start(self, on_message: MessageCallback) -> None:
        """Start the WeChat adapter in webhook mode.

        Messages arrive via HTTP callbacks through the Gateway's API server.
        This method initializes the HTTP client and fetches an access token.
        """
        try:
            import aiohttp
        except ImportError as err:
            raise ImportError(
                "aiohttp is required for WeChat support. Install it with: pip install cliver[wechat]"
            ) from err

        self._on_message = on_message
        self._session = aiohttp.ClientSession()

        # Fetch initial access token
        if self._corp_id and self._token:
            await self._refresh_access_token()

        logger.info("WeChat adapter started (webhook mode)")

    async def stop(self) -> None:
        """Stop the WeChat adapter."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("WeChat adapter stopped")

    # --- Sending ---

    async def send_text(self, channel_id: str, text: str, reply_to: Optional[str] = None) -> None:
        token = await self._get_access_token()
        if not token or not self._session:
            return
        url = f"{WECOM_API_BASE}/message/send?access_token={token}"
        payload = {
            "touser": channel_id,
            "msgtype": "text",
            "agentid": int(self._agent_id) if self._agent_id else 0,
            "text": {"content": text},
        }
        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                logger.warning("WeChat send_text failed: %s", resp.status)

    async def send_image(self, channel_id: str, image: bytes | Path, caption: str = "") -> None:
        token = await self._get_access_token()
        if not token or not self._session:
            return
        # Upload image as temporary media
        media_id = await self._upload_media(image, "image", "image.png")
        if not media_id:
            return
        url = f"{WECOM_API_BASE}/message/send?access_token={token}"
        payload = {
            "touser": channel_id,
            "msgtype": "image",
            "agentid": int(self._agent_id) if self._agent_id else 0,
            "image": {"media_id": media_id},
        }
        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                logger.warning("WeChat send_image failed: %s", resp.status)

        # Send caption as a separate text message if provided
        if caption:
            await self.send_text(channel_id, caption)

    async def send_file(self, channel_id: str, file: bytes | Path, filename: str) -> None:
        token = await self._get_access_token()
        if not token or not self._session:
            return
        media_id = await self._upload_media(file, "file", filename)
        if not media_id:
            return
        url = f"{WECOM_API_BASE}/message/send?access_token={token}"
        payload = {
            "touser": channel_id,
            "msgtype": "file",
            "agentid": int(self._agent_id) if self._agent_id else 0,
            "file": {"media_id": media_id},
        }
        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                logger.warning("WeChat send_file failed: %s", resp.status)

    async def send_voice(self, channel_id: str, audio: bytes | Path) -> None:
        token = await self._get_access_token()
        if not token or not self._session:
            return
        media_id = await self._upload_media(audio, "voice", "voice.amr")
        if not media_id:
            return
        url = f"{WECOM_API_BASE}/message/send?access_token={token}"
        payload = {
            "touser": channel_id,
            "msgtype": "voice",
            "agentid": int(self._agent_id) if self._agent_id else 0,
            "voice": {"media_id": media_id},
        }
        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                logger.warning("WeChat send_voice failed: %s", resp.status)

    async def send_typing(self, channel_id: str) -> None:
        # WeChat/WeCom does not support typing indicators
        pass

    # --- Formatting ---

    def format_message(self, markdown: str) -> str:
        return strip_markdown(markdown)

    def max_message_length(self) -> int:
        return 2048

    # --- Internal helpers ---

    def _is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed (open access if no whitelist)."""
        if self._allowed_users is None:
            return True
        return str(user_id) in self._allowed_users

    async def _refresh_access_token(self) -> None:
        """Fetch a new access token from the WeCom API."""
        if not self._session or not self._corp_id or not self._token:
            return
        url = f"{WECOM_API_BASE}/gettoken?corpid={self._corp_id}&corpsecret={self._token}"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("errcode") == 0:
                        self._access_token = data["access_token"]
                        self._token_expires_at = time.time() + data.get("expires_in", 7200) - 300
                        logger.debug("WeChat access token refreshed")
                    else:
                        logger.warning("WeChat token error: %s", data.get("errmsg"))
        except Exception as e:
            logger.warning("Failed to refresh WeChat access token: %s", e)

    async def _get_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if needed."""
        if not self._access_token or time.time() >= self._token_expires_at:
            await self._refresh_access_token()
        return self._access_token

    async def _upload_media(self, media: bytes | Path, media_type: str, filename: str) -> Optional[str]:
        """Upload temporary media to WeCom and return media_id."""
        token = await self._get_access_token()
        if not token or not self._session:
            return None

        import aiohttp

        url = f"{WECOM_API_BASE}/media/upload?access_token={token}&type={media_type}"
        if isinstance(media, Path):
            with open(media, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("media", f, filename=filename)
                async with self._session.post(url, data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("media_id")
        else:
            data = aiohttp.FormData()
            data.add_field("media", BytesIO(media), filename=filename)
            async with self._session.post(url, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("media_id")

        return None

    async def handle_webhook(self, request_data: dict) -> None:
        """Process an incoming webhook callback from WeCom.

        This method is called by the Gateway's API server when a message
        arrives at the /webhook/wechat endpoint.
        """
        msg_type = request_data.get("MsgType", "")
        user_id = request_data.get("FromUserName", "")

        if not user_id or not self._is_allowed(user_id):
            return

        text = ""
        media = []

        if msg_type == "text":
            text = request_data.get("Content", "")
        elif msg_type == "image":
            pic_url = request_data.get("PicUrl", "")
            if pic_url:
                media.append(MediaAttachment(type="image", url=pic_url))
        elif msg_type == "voice":
            media_id = request_data.get("MediaId", "")
            if media_id:
                media.append(MediaAttachment(type="voice", url=media_id))
        elif msg_type == "file":
            media_id = request_data.get("MediaId", "")
            filename = request_data.get("FileName", "file")
            if media_id:
                media.append(MediaAttachment(type="file", url=media_id, filename=filename))

        event = MessageEvent(
            platform="wechat",
            channel_id=user_id,  # WeCom uses user ID as channel
            user_id=user_id,
            text=text,
            media=media,
            is_group=False,
        )
        if self._on_message:
            await self._on_message(event)
