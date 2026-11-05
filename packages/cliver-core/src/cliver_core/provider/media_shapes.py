"""Wire-shape catalog — the accept/reject authority for reference media.

A :class:`MediaShape` names one request wire: the reference field(s) a
provider fills, the media kinds it accepts, and how many references fit.
The catalog lists wires; it never lists vendors or models — brand modules
declare which shapes they emit, media bindings select one explicitly via
``wire_shape=``, and the model string never influences the choice.

``MediaShape.select`` is the only gate: ``generate_media`` calls it before
any request body is built, and a reference the shape cannot represent
raises :class:`~cliver_core.errors.MediaReferenceError` instead of being
silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cliver_core.errors import MediaReferenceError
from cliver_core.media import MediaType

if TYPE_CHECKING:
    from cliver_core.media import MediaContent


def _kinds(kinds: frozenset[MediaType]) -> str:
    """Accepted kinds as message text — ``image``, ``image, audio``, …"""
    return ", ".join(sorted(kind.value for kind in kinds))


@dataclass(frozen=True)
class MediaShape:
    """One request wire for reference media — the unit of media support.

    * ``accepted`` — reference kinds the wire can carry; empty means the
      wire takes no reference media at all.
    * ``max_references`` — how many references fit the wire's field(s):
      ``None`` is no fixed limit (an array field), ``0`` is none, a number
      is the field count (e.g. one ``image`` field).  It is a property of
      the wire, never of the model — a model with a lower limit than its
      wire lets the vendor reject the request.
    * ``transport`` — documentation for humans and tests: how references
      reach the endpoint (``json``, ``multipart``, ``async_upload``).
    """

    name: str
    purpose: str
    accepted: frozenset[MediaType]
    max_references: int | None
    transport: str

    def select(
        self,
        provider: str,
        purpose: str,
        media: list[MediaContent] | None,
    ) -> list[MediaContent]:
        """Validate *media* against this wire; return the references to encode.

        Returns an empty list when *media* is ``None`` or empty.  Raises
        :class:`MediaReferenceError` for a reference without data, a kind
        the wire does not accept, or more references than the wire's fields
        can carry — always before a request body is built.
        """
        if not media:
            return []
        for index, item in enumerate(media):
            data = getattr(item, "data", None)
            if not isinstance(data, str) or not data:
                raise MediaReferenceError(
                    f"{provider} {purpose}: media[{index}] carries no data — a reference must "
                    "hold a non-empty data string (base64, data: URI, or URL)."
                )
        kinds_got = {item.type for item in media}
        if not kinds_got <= self.accepted:
            raise MediaReferenceError(
                f"{provider} {purpose} uses wire shape '{self.name}', which accepts "
                f"{_kinds(self.accepted) if self.accepted else 'no reference media'}; "
                f"got {_kinds(kinds_got)}. Remove media= or choose a provider/shape that supports it."
            )
        if self.max_references is not None and len(media) > self.max_references:
            label = _kinds(self.accepted) or "reference"
            plural = "" if self.max_references == 1 else "s"
            raise MediaReferenceError(
                f"{provider} {purpose} uses wire shape '{self.name}', which accepts "
                f"{self.max_references} {label} reference{plural}; got {len(media)}. "
                "Remove media= or choose a provider/shape that supports more."
            )
        return list(media)


# ── The catalog ──────────────────────────────────────────────────────────────
#
# One constant per implemented wire.  ASR entries document the multipart
# wires; the ASR path bypasses the body builder, so they are catalog-only.

#: The protocol-default ``{"model", "prompt", ...}`` body — no reference fields.
GENERIC_JSON = MediaShape("generic_json", "any", frozenset(), 0, "json")

OPENAI_IMAGE_JSON = MediaShape("openai_image_json", "image", frozenset({MediaType.IMAGE}), 1, "json")
OPENAI_TTS = MediaShape("openai_tts", "tts", frozenset(), 0, "json")

SEEDANCE_VIDEO = MediaShape("seedance_video", "video", frozenset({MediaType.IMAGE}), 2, "json")

DASHSCOPE_IMAGE_T2I = MediaShape("dashscope_image_t2i", "image", frozenset(), 0, "json")
DASHSCOPE_IMAGE_BACKGROUND = MediaShape(
    "dashscope_image_background", "image", frozenset({MediaType.IMAGE}), 2, "async_upload"
)
DASHSCOPE_VIDEO = MediaShape("dashscope_video", "video", frozenset({MediaType.IMAGE}), 1, "async_upload")

QWEN_IMAGE_MESSAGES = MediaShape("qwen_image_messages", "image", frozenset({MediaType.IMAGE}), None, "json")

MINIMAX_TTS = MediaShape("minimax_tts", "tts", frozenset(), 0, "json")
#: Reference-image support is not live-verified; the shape rejects media until it is.
MINIMAX_IMAGE = MediaShape("minimax_image", "image", frozenset(), 0, "json")

OPENAI_ASR = MediaShape("openai_asr", "asr", frozenset({MediaType.AUDIO}), 1, "multipart")
MINIMAX_ASR = MediaShape("minimax_asr", "asr", frozenset({MediaType.AUDIO}), 1, "multipart")

#: Every implemented wire, by name — binding ``wire_shape=`` values come from here.
SHAPES: dict[str, MediaShape] = {
    shape.name: shape
    for shape in (
        GENERIC_JSON,
        OPENAI_IMAGE_JSON,
        OPENAI_TTS,
        SEEDANCE_VIDEO,
        DASHSCOPE_IMAGE_T2I,
        DASHSCOPE_IMAGE_BACKGROUND,
        DASHSCOPE_VIDEO,
        QWEN_IMAGE_MESSAGES,
        MINIMAX_TTS,
        MINIMAX_IMAGE,
        OPENAI_ASR,
        MINIMAX_ASR,
    )
}
