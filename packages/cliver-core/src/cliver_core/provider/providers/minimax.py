"""MiniMax: Anthropic-compatible chat endpoint, plus image/TTS on MiniMax's
own media endpoints.

Chat is fully native through the official Anthropic SDK: tool calls arrive
as standard ``tool_use`` blocks — never as XML — and ``/anthropic`` streams
native ``thinking`` blocks (enable thinking in the request).  The
brand-specific surface is media: image/TTS have their own endpoints
(request bodies and response shapes), on the platform's ``/v1`` root.
"""

from __future__ import annotations

import base64

from cliver_core.provider import Provider
from cliver_core.provider.providers.anthropic import AnthropicProvider


class _MiniMaxBase(Provider):
    """MiniMax media quirks shared by the brand's leaf (mixin base, no
    protocol lock).

    Chat, tools, and thinking are fully native through the generic protocol
    provider — no option filtering, response rewriting, or chunk remapping.
    Image/TTS are served on MiniMax's own media domain (brand URLs, body,
    and response shapes) through the shared base media pipeline.
    """

    protocol = ""  # intermediate opt-out — leaves lock the protocol
    brand = "minimax"

    # MiniMax media endpoints (image, TTS) live on MiniMax's own domain.
    # These are the only media endpoints MiniMax serves, so no wire-standard
    # fallback applies — other purposes report unsupported.
    _MEDIA_GEN_BASE = "https://api.minimaxi.com/v1"
    _media_gen_urls: dict[str, str] = {
        "image": "/image_generation",
        "tts": "/t2a_v2",
    }
    _standard_media_fallback = False

    #: T2A requires ``voice_setting.voice_id`` — omitting it fails with
    #: base_resp 2013 "invalid params, empty field".  The default mirrors
    #: OpenAIProvider's ``voice="alloy"``.
    _DEFAULT_TTS_VOICE = "English_expressive_narrator"

    #: TTS options that fold into ``voice_setting`` / ``audio_setting``
    #: instead of riding the request body's top level.
    _TTS_VOICE_TUNING = ("speed", "vol", "pitch", "emotion", "text_normalization", "latex_read")
    _TTS_AUDIO_TUNING = ("sample_rate", "bitrate", "channel")

    def _media_gen_base(self) -> str:
        """The base URL media POSTs resolve against.

        The configured ``base_url`` wins, so a proxy or the other MiniMax
        platform (``api.minimax.io``) retargets image/TTS too.  The chat
        base is anthropic-shaped (``…/anthropic``) — its media lives on the
        sibling ``…/v1`` root, the same one the brand default names.
        """
        base = (self.base_url or self._MEDIA_GEN_BASE).rstrip("/")
        if base.endswith("/anthropic"):
            return base[: -len("/anthropic")] + "/v1"
        return base

    def _resolve_media_gen_url(self, purpose: str, override: str | None = None) -> str | None:
        path = self.media_path(purpose, override)
        if path is None:
            return None
        if path.startswith(("http://", "https://")):
            return path  # absolute override — use as-is
        return self._media_gen_base().rstrip("/") + path

    def _build_media_gen_body(
        self,
        prompt: str,
        model: str,
        purpose: str,
        *,
        media: list | None = None,
        **options,
    ) -> dict:
        """Build the request body — MiniMax's T2A v2 shape for ``tts``.

        The shared pipeline passes the text as *prompt*; T2A calls it
        ``text``.  ``voice`` → ``voice_setting.voice_id`` (always sent — the
        vendor requires it), ``response_format`` → ``audio_setting.format``,
        tuning options fold into the two settings objects, and the audio is
        requested hex-encoded (MiniMax returns bytes as hex, not base64).
        Other purposes use the default body.
        """
        if purpose != "tts":
            return super()._build_media_gen_body(prompt, model, purpose, media=media, **options)

        voice_setting: dict = {"voice_id": options.pop("voice", None) or self._DEFAULT_TTS_VOICE}
        audio_setting: dict = {"format": options.pop("response_format", None) or "mp3"}
        for key in self._TTS_VOICE_TUNING:
            if key in options:
                voice_setting[key] = options.pop(key)
        for key in self._TTS_AUDIO_TUNING:
            if key in options:
                audio_setting[key] = options.pop(key)

        return {
            "model": model,
            "text": prompt,
            "output_format": "hex",
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            **options,
        }

    def _check_media_response_error(self, data: dict, purpose: str) -> str | None:
        """MiniMax wraps errors in ``base_resp`` even on HTTP 200."""
        base = data.get("base_resp", {})
        if isinstance(base, dict) and base.get("status_code", 0) != 0:
            return base.get("status_msg", f"error code {base.get('status_code')}")
        if purpose == "tts":
            # A syntactically successful envelope can still carry no audio —
            # surface that as an error instead of a silent "Generated 0".
            inner = data.get("data")
            audio = inner.get("audio") if isinstance(inner, dict) else None
            if not isinstance(audio, str) or not audio:
                return "response carried no audio payload"
            try:
                bytes.fromhex(audio)
            except ValueError:
                return "audio payload is not valid hex"
        return None

    def _extract_media_items(self, data: dict, purpose: str) -> list[dict]:
        """MiniMax returns images in ``image_urls``; TTS audio hex in ``data.audio``."""
        if purpose == "tts":
            inner = data.get("data")
            audio = inner.get("audio") if isinstance(inner, dict) else None
            if not isinstance(audio, str) or not audio:
                return []
            return [{"b64_json": base64.b64encode(bytes.fromhex(audio)).decode()}]
        # Try nested: data.data.image_urls
        inner = data.get("data", data)
        image_urls = inner.get("image_urls") or data.get("image_urls") or []
        return [{"url": u} for u in image_urls if isinstance(u, str)]


class MiniMaxAnthropicProvider(_MiniMaxBase, AnthropicProvider):
    """MiniMax Anthropic-compatible endpoint — the brand's chat leaf.

    Uses Bearer auth (same as OpenAI), not Anthropic's native x-api-key
    header.
    """

    protocol = "anthropic"  # explicit: the mixin's "" opt-out precedes the base in MRO
    default_base_url = "https://api.minimaxi.com/anthropic"
    _anthropic_use_bearer_auth = True
