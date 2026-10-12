"""Seedance (Volcano Engine Ark): Doubao video generation as async tasks.

Video generation is asynchronous: ``POST /contents/generations/tasks``
answers a task id (``{"id": "cgt-..."}``), and the finished video appears in
the task object's ``content.video_url`` once ``GET
/contents/generations/tasks/{id}`` reports ``status`` ``succeeded``.
:meth:`SeedanceProvider._await_media_result` polls until the task is
terminal; the shared media pipeline then extracts, labels, and saves the
MP4 as usual.  The rest of Ark's ``/api/v3`` is OpenAI-compatible, so
chat/stream ride the bundled OpenAI-SDK transport unchanged.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from cliver_core.media import MediaContent, MediaType
from cliver_core.provider.providers.openai import OpenAIProvider


async def _sleep(seconds: float) -> None:
    """One poll-cadence sleep — a single seam tests can stub out."""
    await asyncio.sleep(seconds)


class SeedanceProvider(OpenAIProvider):
    """Doubao Seedance video on Volcano Engine Ark (bearer auth).

    Media surface: video only — the async create-and-poll wire.  Other media
    purposes report unsupported (Ark has no OpenAI-fashion media endpoints).
    Poll knobs ride the request options but never the body: ``poll_interval``
    (seconds between task reads, default 5) and ``poll_timeout`` (seconds to
    wait for a terminal task, default 600; ``None`` waits without a
    deadline).  Chat on Ark's ``/api/v3`` is OpenAI-compatible and comes
    from the openai protocol unchanged.
    """

    brand = "seedance"
    default_base_url = "https://ark.cn-beijing.volces.com/api/v3"

    #: The only media endpoint Ark serves — task-shaped, not OpenAI-fashion.
    _media_gen_urls: dict[str, str] = {"video": "/contents/generations/tasks"}
    _standard_media_fallback = False

    #: Create → poll cadence.
    _POLL_INTERVAL = 5.0
    _POLL_TIMEOUT = 600.0
    _POLL_REQUEST_TIMEOUT = 30.0

    #: Task states that still move toward a result; everything else terminal.
    _PENDING_TASK_STATUSES = frozenset({"queued", "running"})

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
        """Ark's task-submission body for ``video``: a ``content`` array.

        The prompt is a ``text`` item; reference images (image-to-video) are
        ``image_url`` items.  Poll knobs are this brand's request options,
        not Ark body fields — dropped from this builder's own copy (the
        await hook reads the original options dict).
        """
        if purpose != "video":
            return super()._build_media_gen_body(prompt, model, purpose, media=media, **options)
        options.pop("poll_interval", None)
        options.pop("poll_timeout", None)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in media or []:
            if getattr(item, "type", None) != MediaType.IMAGE:
                continue
            image = item.data
            if not image.startswith("data:"):
                image = f"data:{item.mime_type};base64,{image}"
            content.append({"type": "image_url", "image_url": {"url": image}})
        return {"model": model, "content": content, **options}

    # ── async task resolution ────────────────────────────────────

    async def _await_media_result(self, raw: Any, purpose: str, url: str, options: dict[str, Any]) -> Any:
        """Poll the created task until terminal; return the finished task.

        The deadline is checked **before** every task read, so
        ``poll_timeout=0`` performs zero GETs and fails deterministically.
        Only ``succeeded`` (with a video URL) is a result; ``failed`` /
        ``cancelled`` / ``expired`` / unknown states raise with the vendor's
        error message and land in the shared friendly-failure path.
        """
        if purpose != "video":
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raise RuntimeError("binary=True is not supported for seedance video — the endpoint submits an async task")
        if not isinstance(raw, dict) or not raw.get("id"):
            raise RuntimeError(f"seedance video create returned no task id: {raw!r:.300}")

        task_id = str(raw["id"])
        interval = options.get("poll_interval", self._POLL_INTERVAL)
        timeout = options.get("poll_timeout", self._POLL_TIMEOUT)
        deadline = None if timeout is None else time.monotonic() + timeout

        payload = raw
        while True:
            status = payload.get("status")
            if status == "succeeded":
                return self._finished_task(payload, task_id)
            if status is None or status in self._PENDING_TASK_STATUSES:
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"seedance video task {task_id} timed out after {timeout}s (last status: {status or 'unknown'})"
                    )
                await _sleep(interval)
                payload = await self._fetch_task(task_id, url=url)
                continue
            raise RuntimeError(f"seedance video task {task_id} {status}: {self._task_error(payload)}")

    @staticmethod
    def _finished_task(payload: dict, task_id: str) -> dict:
        """A succeeded task must carry the video URL — else it is a failure.

        Without this guard the pipeline would report a silent
        "Generated 0 video(s)" success for a payload it cannot use.
        """
        content = payload.get("content")
        video_url = content.get("video_url") if isinstance(content, dict) else None
        if not isinstance(video_url, str) or not video_url:
            raise RuntimeError(f"seedance video task {task_id} succeeded without a content.video_url")
        return payload

    @staticmethod
    def _task_error(payload: dict) -> str:
        """The vendor's error message (dict or string), else a payload excerpt."""
        error = payload.get("error")
        if isinstance(error, dict):
            error = error.get("message") or error.get("code")
        if isinstance(error, str) and error:
            return error
        return str(payload)[:300]

    async def _fetch_task(self, task_id: str, *, url: str) -> dict:
        """Read one task object — a single GET, no retries.

        The task URL derives from *url*, the create endpoint exactly as
        POSTed, so a call-site ``path=`` (proxy, custom host) is honored.
        The transport protocol has no GET verb and extending it would break
        third-party adapters, so this rides httpx directly — the same
        sessionless shape as the embed/rerank precedent
        (``Provider._post_json_endpoint``).
        """
        async with httpx.AsyncClient(timeout=self._POLL_REQUEST_TIMEOUT) as client:
            response = await client.get(f"{url}/{task_id}", headers=self._build_media_gen_headers())
        if not response.is_success:
            raise RuntimeError(f"seedance task read failed: HTTP {response.status_code} {response.text[:200]}")
        return response.json()

    def _extract_media_items(self, data: dict, purpose: str) -> list[dict]:
        """The finished task carries the video in ``content.video_url``."""
        if purpose == "video" and isinstance(data, dict):
            content = data.get("content")
            video_url = content.get("video_url") if isinstance(content, dict) else None
            if isinstance(video_url, str) and video_url:
                return [{"url": video_url}]
        return super()._extract_media_items(data, purpose)
