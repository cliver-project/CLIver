"""Qwen-Image — DashScope's synchronous reference-aware image wire.

Qwen-Image models (``qwen-image-3.0``, ``wan2.6-image``, …) serve on
``POST /api/v1/services/aigc/multimodal-generation/generation``: one user
message whose content alternates reference images and the text instruction,
answered in the same request with the generated image URL in
``output.choices[*].message.content[*].image``.  No task id, no polling.

Unlike the async text-to-image wire, this endpoint accepts ``data:`` URIs
directly, so local reference bytes need no upload — the shared
:meth:`DashScopeProvider._image_values` encodes them and
:meth:`DashScopeProvider._prepare_media_body` (which only rewrites the async
wire's top-level ``input`` fields) leaves the message content untouched.
Live-verified 2026-10-29 on the classic DashScope host.
"""

from __future__ import annotations

from typing import Any

from cliver_core.media import MediaContent
from cliver_core.provider.providers.dashscope import DashScopeProvider


class QwenImageProvider(DashScopeProvider):
    """Qwen-Image generation (bearer auth, synchronous result).

    Media surface: ``image`` only.  The prompt is the instruction
    (``input.messages[0].content[].text``); each IMAGE media item becomes
    an ``{"image": ...}`` content part, in order — public ``http(s)`` and
    ``oss://`` URLs pass through, local bytes ride as ``data:`` URIs.
    Every other option (``size``, ``n``, ``negative_prompt``,
    ``prompt_extend``, ``watermark``, ``seed`` …) folds into the body's
    ``parameters``; sizes normalize to DashScope's ``*`` separator.
    """

    brand = "qwen-image"
    default_base_url = "https://dashscope.aliyuncs.com/api/v1"

    _media_gen_urls: dict[str, str] = {
        "image": "/services/aigc/multimodal-generation/generation",
    }
    _standard_media_fallback = False

    # ── create body ──────────────────────────────────────────────

    def _build_media_gen_body(
        self,
        prompt: str,
        model: str,
        purpose: str,
        *,
        media: list[MediaContent] | None = None,
        **options,
    ) -> dict[str, Any]:
        """One multimodal user message: reference images, then the instruction."""
        if purpose != "image":
            return super()._build_media_gen_body(prompt, model, purpose, media=media, **options)

        params = {name: value for name, value in options.items() if name not in self._POLL_OPTIONS}
        size = params.get("size")
        if isinstance(size, str):
            params["size"] = size.replace("x", "*").replace("X", "*")

        content: list[dict[str, Any]] = [{"image": value} for value in self._image_values(media)]
        content.append({"text": prompt})
        return {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": params,
        }

    # ── synchronous result ───────────────────────────────────────

    async def _await_media_result(self, raw: Any, purpose: str, url: str, options: dict[str, Any]) -> Any:
        """Validate the synchronous payload; there is no task to poll.

        A result must carry at least one image URL — without this guard the
        pipeline would report a silent "Generated 0 image(s)" for a payload
        it cannot use.  HTTP-200 error envelopes surface with the vendor's
        message.
        """
        if purpose != "image":
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raise RuntimeError("binary=True is not supported for qwen-image media — the endpoint returns a JSON result")
        if not isinstance(raw, dict):
            raise RuntimeError(f"qwen-image {purpose} returned a non-JSON payload: {raw!r:.300}")
        envelope = self._check_media_response_error(raw, purpose)
        if envelope:
            raise RuntimeError(f"qwen-image {purpose} failed: {envelope}")
        if not self._image_urls(raw):
            raise RuntimeError(f"qwen-image {purpose} returned no image: {raw!r:.300}")
        return raw

    @staticmethod
    def _image_urls(data: dict) -> list[str]:
        """Every generated image URL, in ``output.choices[*].message.content`` order."""
        output = data.get("output") if isinstance(data, dict) else None
        choices = output.get("choices") if isinstance(output, dict) else None
        urls: list[str] = []
        for choice in choices if isinstance(choices, list) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and isinstance(part.get("image"), str) and part["image"]:
                    urls.append(part["image"])
        return urls

    def _extract_media_items(self, data: dict, purpose: str) -> list[dict]:
        urls = self._image_urls(data) if purpose == "image" else []
        if urls:
            return [{"url": url} for url in urls]
        return super()._extract_media_items(data, purpose)

    # ── headers ──────────────────────────────────────────────────

    def _media_request_headers(self, purpose: str, body: dict[str, Any]) -> dict[str, str]:
        """Synchronous call — no async flag; the OSS resolver only when
        the body actually carries an ``oss://`` URL."""
        if self._contains_oss_url(body):
            return {"X-DashScope-OssResourceResolve": "enable"}
        return {}
