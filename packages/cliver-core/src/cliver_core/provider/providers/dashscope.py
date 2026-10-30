"""DashScope (Alibaba Model Studio / Bailian): async media tasks on ``/api/v1``.

Every DashScope media call is asynchronous.  The create POST (with
``X-DashScope-Async: enable``) answers ``{"output": {"task_id": ...,
"task_status": "PENDING"}}``; the result only exists once ``GET
/api/v1/tasks/{id}`` reports ``SUCCEEDED`` — image URLs in
``output.results[*].url``, video in ``output.video_url``.
:meth:`DashScopeProvider._await_media_result` polls until the task is
terminal; the shared media pipeline then extracts, labels, saves, and (for
``output_dir``) downloads the result — these URLs expire after 24 h.

Reference images are accepted only as public ``http(s)`` URLs or DashScope
``oss://`` temporary URLs.  Local bytes ride the usual ``media=[...]``
surface: :meth:`DashScopeProvider._prepare_media_body` uploads every
data-URI image through :meth:`DashScopeProvider.upload_media` (policy GET →
OSS multipart POST) and rewrites the body to the returned ``oss://`` URL,
adding the ``X-DashScope-OssResourceResolve`` header.  Chat on the
OpenAI-compatible endpoint is the separate ``qwen`` brand.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import httpx

from cliver_core.errors import ProviderError
from cliver_core.media import MediaContent, MediaType, get_file_extension
from cliver_core.provider.providers.openai import OpenAIProvider


async def _sleep(seconds: float) -> None:
    """One poll-cadence sleep — a single seam tests can stub out."""
    await asyncio.sleep(seconds)


class DashScopeProvider(OpenAIProvider):
    """Alibaba Model Studio media generation (bearer auth).

    Media surface: ``image`` (text-to-image, plus background generation when
    the model name selects it) and ``video`` — both on the async
    create-and-poll wire.  Other media purposes report unsupported
    (DashScope has no OpenAI-fashion media endpoints).  Poll knobs ride the
    request options but never the body: ``poll_interval`` (seconds between
    task reads, default 3) and ``poll_timeout`` (seconds to wait for a
    terminal task, default 600; ``None`` waits without a deadline).
    """

    brand = "dashscope"
    default_base_url = "https://dashscope.aliyuncs.com/api/v1"

    _media_gen_urls: dict[str, str] = {
        "image": "/services/aigc/text2image/image-synthesis",
        "video": "/services/aigc/video-generation/video-synthesis",
    }
    _standard_media_fallback = False

    #: Background generation shares the ``image`` purpose but has its own
    #: endpoint (pinned by the ``dashscope_background`` preset) and body
    #: shape; the model name is what selects it.
    _BACKGROUND_MODEL_PREFIX = "wanx-background-generation"

    #: The ``input`` fields that carry a reference image, in preference order.
    _INPUT_IMAGE_FIELDS = ("base_image_url", "img_url", "ref_image_url")

    #: Create → poll cadence.
    _POLL_INTERVAL = 3.0
    _POLL_TIMEOUT = 600.0
    _POLL_REQUEST_TIMEOUT = 30.0
    _UPLOAD_TIMEOUT = 120.0

    #: Task states that still move toward a result; everything else terminal.
    _PENDING_TASK_STATUSES = frozenset({"PENDING", "RUNNING"})

    #: Request options owned by the poll loop — never body fields.
    _POLL_OPTIONS = ("poll_interval", "poll_timeout")

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
        """DashScope's ``{"model", "input", "parameters"}`` request shape.

        The prompt lands in ``input.prompt`` (text-to-image, video) or
        ``input.ref_prompt`` (background generation, selected by the model
        name).  The first reference image feeds ``input.base_image_url``
        (background) or ``input.img_url`` (image-to-video); a second
        background reference feeds ``input.ref_image_url``.  Every other
        option folds into ``parameters``, ``negative_prompt`` rides
        ``input``, and sizes normalize to DashScope's ``*`` separator.
        Poll knobs are request options, not body fields.
        """
        if purpose not in ("image", "video"):
            return super()._build_media_gen_body(prompt, model, purpose, media=media, **options)

        params = {name: value for name, value in options.items() if name not in self._POLL_OPTIONS}
        size = params.get("size")
        if isinstance(size, str):
            params["size"] = size.replace("x", "*").replace("X", "*")
        negative_prompt = params.pop("negative_prompt", None)

        inputs: dict[str, Any] = {}
        if purpose == "video":
            inputs["prompt"] = prompt
            images = self._image_values(media)
            if images:
                inputs["img_url"] = images[0]
        elif model.startswith(self._BACKGROUND_MODEL_PREFIX):
            inputs["ref_prompt"] = prompt
            images = self._image_values(media)
            if images:
                inputs["base_image_url"] = images[0]
            if len(images) > 1:
                inputs["ref_image_url"] = images[1]
        else:
            inputs["prompt"] = prompt
        if negative_prompt is not None:
            inputs["negative_prompt"] = negative_prompt

        return {"model": model, "input": inputs, "parameters": params}

    @staticmethod
    def _image_values(media: list[MediaContent] | None) -> list[str]:
        """Reference images as the strings the body will carry before upload.

        Public and ``oss://`` URLs pass through; local bytes become a
        ``data:<mime>;base64,...`` URI — self-describing, so
        :meth:`_prepare_media_body` can decode it without the media item.
        """
        values: list[str] = []
        for item in media or []:
            if getattr(item, "type", None) != MediaType.IMAGE:
                continue
            data = getattr(item, "data", "")
            if not isinstance(data, str) or not data:
                continue
            if data.startswith(("data:", "http://", "https://", "oss://")):
                values.append(data)
            else:
                mime = getattr(item, "mime_type", "") or "image/png"
                values.append(f"data:{mime};base64,{data}")
        return values

    # ── local reference images → oss:// ─────────────────────────

    async def _prepare_media_body(self, body: dict[str, Any], purpose: str, *, model: str) -> dict[str, Any]:
        """Upload local reference images and rewrite their fields to ``oss://``.

        DashScope accepts reference images only as public ``http(s)`` or
        temporary ``oss://`` URLs (never a data URI).  Each data-URI value in
        the body's image inputs is uploaded once and replaced; public URLs
        pass through untouched.
        """
        inputs = body.get("input") if isinstance(body, dict) else None
        if not isinstance(inputs, dict):
            return body
        uploads = 0
        for field in self._INPUT_IMAGE_FIELDS:
            value = inputs.get(field)
            if not isinstance(value, str) or not value:
                continue
            if value.startswith(("http://", "https://", "oss://")):
                continue
            data, mime = self._decode_data_uri(value)
            filename = f"reference_{uploads}{get_file_extension(mime)}"
            inputs[field] = await self.upload_media(data, mime, model=model, filename=filename)
            uploads += 1
        return body

    @staticmethod
    def _decode_data_uri(value: str) -> tuple[bytes, str]:
        """Decode one ``data:<mime>;base64,<payload>`` URI into bytes + mime."""
        if not value.startswith("data:"):
            raise ProviderError(
                "dashscope reference images must be a public http(s) URL, an oss:// URL, "
                f"or base64 bytes — got {value[:40]!r}"
            )
        header, _, payload = value.partition(",")
        if not payload:
            raise ProviderError("dashscope reference image is an empty data URI")
        mime, _, encoding = header[len("data:") :].partition(";")
        if encoding.lower() != "base64":
            raise ProviderError(
                f"dashscope reference image data URI must be base64-encoded (got '{encoding or 'plain'}')"
            )
        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError) as exc:
            raise ProviderError(f"dashscope reference image data URI is not valid base64: {exc}") from exc
        return data, mime or "application/octet-stream"

    async def upload_media(self, data: bytes, mime: str, *, model: str, filename: str) -> str:
        """Upload local bytes to DashScope temporary storage; return an ``oss://`` URL.

        Two calls: ``GET /uploads?action=getPolicy&model=<model>`` for an
        upload policy, then one multipart POST to the policy's
        ``upload_host`` (``file`` last, as OSS requires).  The returned URL
        stays valid for about 48 hours — enough for the task that consumes
        it.  Failures raise :class:`~cliver_core.errors.ProviderError` with
        the vendor's status and body excerpt; nothing fails silently.
        """
        if not data:
            raise ProviderError("dashscope upload requires non-empty bytes")
        if not filename:
            raise ProviderError("dashscope upload requires a filename")

        policy = await self._upload_policy(model)
        missing = [
            name
            for name in ("upload_host", "upload_dir", "policy", "signature", "oss_access_key_id")
            if not policy.get(name)
        ]
        if missing:
            raise ProviderError(f"dashscope upload policy is missing {', '.join(missing)}")

        upload_dir = str(policy["upload_dir"])
        key = f"{upload_dir}/{filename}"
        fields = {
            "OSSAccessKeyId": policy["oss_access_key_id"],
            "policy": policy["policy"],
            "Signature": policy["signature"],
            "key": key,
            "x-oss-object-acl": str(policy.get("x_oss_object_acl") or "private"),
            "x-oss-forbid-overwrite": str(policy.get("x_oss_forbid_overwrite") or "true"),
            "success_action_status": "200",
        }
        async with httpx.AsyncClient(timeout=self._UPLOAD_TIMEOUT) as client:
            response = await client.post(
                str(policy["upload_host"]),
                data=fields,
                files={"file": (filename, data, mime)},
            )
        if not response.is_success:
            raise ProviderError(f"dashscope upload failed: HTTP {response.status_code} {response.text[:200]}")
        return f"oss://{key}"

    async def _upload_policy(self, model: str) -> dict[str, Any]:
        """One policy read — the ``data`` object of ``GET /uploads``."""
        url = f"{self.base_url.rstrip('/')}/uploads"
        async with httpx.AsyncClient(timeout=self._POLL_REQUEST_TIMEOUT) as client:
            response = await client.get(
                url,
                params={"action": "getPolicy", "model": model},
                headers=self._build_media_gen_headers(),
            )
        if not response.is_success:
            raise ProviderError(
                f"dashscope upload policy request failed: HTTP {response.status_code} {response.text[:200]}"
            )
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise ProviderError(f"dashscope upload policy response has no data: {payload!r:.300}")
        return data

    # ── async task resolution ────────────────────────────────────

    async def _await_media_result(self, raw: Any, purpose: str, url: str, options: dict[str, Any]) -> Any:
        """Poll the created task until terminal; return the finished task.

        The deadline is checked **before** every task read, so
        ``poll_timeout=0`` performs zero GETs and fails deterministically.
        Only ``SUCCEEDED`` with a usable URL is a result; ``FAILED`` /
        ``CANCELED`` / ``UNKNOWN`` raise with the vendor's message and land
        in the shared friendly-failure path.
        """
        if purpose not in ("image", "video"):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raise RuntimeError("binary=True is not supported for dashscope media — the endpoint submits an async task")
        if not isinstance(raw, dict):
            raise RuntimeError(f"dashscope {purpose} create returned a non-JSON payload: {raw!r:.300}")

        task_id = self._task_id(raw)
        if not task_id:
            envelope = self._check_media_response_error(raw, purpose)
            if envelope:
                raise RuntimeError(f"dashscope {purpose} create failed: {envelope}")
            raise RuntimeError(f"dashscope {purpose} create returned no task id: {raw!r:.300}")

        interval = options.get("poll_interval", self._POLL_INTERVAL)
        timeout = options.get("poll_timeout", self._POLL_TIMEOUT)
        deadline = None if timeout is None else time.monotonic() + timeout

        payload = raw
        while True:
            output = payload.get("output") if isinstance(payload, dict) else None
            status = output.get("task_status") if isinstance(output, dict) else None
            if status == "SUCCEEDED":
                return self._finished_task(payload, task_id, purpose)
            if status is None or status in self._PENDING_TASK_STATUSES:
                if status is None:
                    envelope = self._check_media_response_error(payload, purpose)
                    if envelope:
                        raise RuntimeError(f"dashscope {purpose} task {task_id} read failed: {envelope}")
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"dashscope {purpose} task {task_id} timed out after {timeout}s "
                        f"(last status: {status or 'unknown'})"
                    )
                await _sleep(interval)
                payload = await self._fetch_task(task_id, url=url)
                continue
            raise RuntimeError(f"dashscope {purpose} task {task_id} {status}: {self._task_error(payload)}")

    @staticmethod
    def _task_id(payload: dict) -> str | None:
        output = payload.get("output")
        task_id = output.get("task_id") if isinstance(output, dict) else None
        return str(task_id) if task_id else None

    def _finished_task(self, payload: dict, task_id: str, purpose: str) -> dict:
        """A succeeded task must carry a usable URL — else it is a failure.

        Without this guard the pipeline would report a silent
        "Generated 0 image(s)" success for a payload it cannot use.
        """
        output = payload.get("output")
        if not isinstance(output, dict):
            output = {}
        if purpose == "video":
            video_url = output.get("video_url")
            if not isinstance(video_url, str) or not video_url:
                raise RuntimeError(f"dashscope video task {task_id} succeeded without an output.video_url")
            return payload

        results = output.get("results")
        urls = (
            [
                result["url"]
                for result in results
                if isinstance(result, dict) and isinstance(result.get("url"), str) and result["url"]
            ]
            if isinstance(results, list)
            else []
        )
        if not urls:
            detail = self._result_errors(results)
            suffix = f" ({detail})" if detail else ""
            raise RuntimeError(f"dashscope image task {task_id} succeeded without a result url{suffix}")
        return payload

    @staticmethod
    def _result_errors(results: Any) -> str:
        """Vendor error lines from per-result ``code``/``message`` fields."""
        messages: list[str] = []
        for result in results if isinstance(results, list) else []:
            if not isinstance(result, dict):
                continue
            code = result.get("code")
            message = result.get("message")
            line = ": ".join(str(part) for part in (code, message) if part)
            if line:
                messages.append(line)
        return "; ".join(messages)

    @staticmethod
    def _task_error(payload: dict) -> str:
        """The vendor's task error message, else a payload excerpt."""
        output = payload.get("output") if isinstance(payload, dict) else None
        if isinstance(output, dict):
            message = output.get("message") or output.get("code")
            if isinstance(message, str) and message:
                return message
        return str(payload)[:300]

    def _task_url(self, task_id: str, create_url: str) -> str:
        """The task read URL — ``{api root}/tasks/{id}``, not create + id.

        DashScope's create routes end in ``/generation`` while the task
        route hangs off the API root.  The root derives from the create URL
        when it carries the ``/services/`` marker (so a call-site ``path=``
        onto a proxy keeps the proxy), and from ``base_url`` otherwise.
        """
        root = self.base_url or ""
        if "/services/" in create_url:
            root = create_url.split("/services/", 1)[0]
        return f"{root.rstrip('/')}/tasks/{task_id}"

    async def _fetch_task(self, task_id: str, *, url: str) -> dict:
        """Read one task object — a single GET, no retries.

        The transport protocol has no GET verb, so this rides httpx directly
        — the same sessionless shape as the embed/rerank precedent
        (``Provider._post_json_endpoint``).
        """
        async with httpx.AsyncClient(timeout=self._POLL_REQUEST_TIMEOUT) as client:
            response = await client.get(self._task_url(task_id, url), headers=self._build_media_gen_headers())
        if not response.is_success:
            raise RuntimeError(f"dashscope task read failed: HTTP {response.status_code} {response.text[:200]}")
        return response.json()

    # ── headers + response shapes ────────────────────────────────

    def _media_request_headers(self, purpose: str, body: dict[str, Any]) -> dict[str, str]:
        """Create-call headers: every DashScope media call is asynchronous.

        The OSS resolver flag goes out only when the body actually carries an
        ``oss://`` URL (a local reference image was just uploaded).
        """
        headers = {"X-DashScope-Async": "enable"}
        if self._contains_oss_url(body):
            headers["X-DashScope-OssResourceResolve"] = "enable"
        return headers

    @classmethod
    def _contains_oss_url(cls, value: Any) -> bool:
        if isinstance(value, str):
            return "oss://" in value
        if isinstance(value, dict):
            return any(cls._contains_oss_url(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._contains_oss_url(item) for item in value)
        return False

    def _check_media_response_error(self, data: dict, purpose: str) -> str | None:
        """DashScope's HTTP-200 error envelope is a top-level ``code``/``message``."""
        if isinstance(data, dict):
            code = data.get("code")
            message = data.get("message")
            if code:
                return f"{code}: {message}" if message else str(code)
        return None

    def _extract_media_items(self, data: dict, purpose: str) -> list[dict]:
        """Finished tasks carry image URLs in ``output.results`` and video in
        ``output.video_url``."""
        if isinstance(data, dict):
            output = data.get("output")
            if isinstance(output, dict):
                if purpose == "video":
                    video_url = output.get("video_url")
                    if isinstance(video_url, str) and video_url:
                        return [{"url": video_url}]
                results = output.get("results")
                if isinstance(results, list):
                    return [
                        {"url": result["url"]}
                        for result in results
                        if isinstance(result, dict) and isinstance(result.get("url"), str) and result["url"]
                    ]
        return super()._extract_media_items(data, purpose)
