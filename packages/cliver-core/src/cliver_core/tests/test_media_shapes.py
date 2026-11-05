"""Wire-shape catalog — the accept/reject authority for reference media.

A ``MediaShape`` names one request wire (endpoint shape, reference fields,
accepted kinds, reference limit).  ``select`` is the only gate: it validates
the caller's references against the shape before any request body is built.
"""

from __future__ import annotations

import pytest

from cliver_core import MediaReferenceError
from cliver_core.errors import MediaReferenceError as ErrorsModuleMediaReferenceError
from cliver_core.media import MediaContent, MediaType
from cliver_core.provider.media_shapes import (
    DASHSCOPE_IMAGE_BACKGROUND,
    DASHSCOPE_IMAGE_T2I,
    DASHSCOPE_VIDEO,
    GENERIC_JSON,
    MINIMAX_ASR,
    MINIMAX_IMAGE,
    MINIMAX_TTS,
    OPENAI_ASR,
    OPENAI_IMAGE_JSON,
    OPENAI_TTS,
    QWEN_IMAGE_MESSAGES,
    SEEDANCE_VIDEO,
    SHAPES,
    MediaShape,
)


def img(index: int) -> MediaContent:
    return MediaContent(type=MediaType.IMAGE, data=f"data:image/png;base64,IMG{index}", mime_type="image/png")


def audio(index: int) -> MediaContent:
    return MediaContent(type=MediaType.AUDIO, data=f"data:audio/mpeg;base64,AUD{index}", mime_type="audio/mpeg")


def test_media_reference_error_is_a_value_error() -> None:
    assert issubclass(MediaReferenceError, ValueError)
    # The same class under both import paths — cliver_core re-exports it.
    assert MediaReferenceError is ErrorsModuleMediaReferenceError


def test_select_returns_empty_for_none_and_empty_list() -> None:
    assert OPENAI_IMAGE_JSON.select("openai", "image", None) == []
    assert GENERIC_JSON.select("any-provider", "image", []) == []


def test_select_passes_allowed_kinds_within_limit() -> None:
    two = [img(0), img(1)]
    assert DASHSCOPE_IMAGE_BACKGROUND.select("dashscope", "image", two) == two
    # No fixed limit — every reference passes.
    three = [img(0), img(1), img(2)]
    assert QWEN_IMAGE_MESSAGES.select("qwen-image", "image", three) == three


def test_select_rejects_unsupported_kind() -> None:
    with pytest.raises(MediaReferenceError, match="audio"):
        OPENAI_IMAGE_JSON.select("openai", "image", [audio(0)])


def test_select_rejects_over_limit_count() -> None:
    with pytest.raises(MediaReferenceError, match="got 2"):
        OPENAI_IMAGE_JSON.select("openai", "image", [img(0), img(1)])


def test_select_rejects_any_media_on_zero_shapes() -> None:
    with pytest.raises(MediaReferenceError, match="no reference media"):
        GENERIC_JSON.select("some-brand", "image", [img(0)])
    with pytest.raises(MediaReferenceError, match="no reference media"):
        MINIMAX_TTS.select("minimax", "tts", [audio(0)])


def test_select_rejects_reference_without_data() -> None:
    empty = MediaContent(type=MediaType.IMAGE, data="", mime_type="image/png")
    with pytest.raises(MediaReferenceError, match="no data"):
        OPENAI_IMAGE_JSON.select("openai", "image", [empty])


def test_error_message_names_provider_purpose_shape_and_limit() -> None:
    with pytest.raises(MediaReferenceError) as excinfo:
        OPENAI_IMAGE_JSON.select("openai", "image", [img(0), img(1)])
    message = str(excinfo.value)
    assert "openai" in message
    assert "image" in message
    assert "openai_image_json" in message
    assert "1 image reference" in message


def test_catalog_registers_every_constant_under_its_name() -> None:
    constants = [
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
    ]
    for shape in constants:
        assert isinstance(shape, MediaShape)
        assert SHAPES[shape.name] is shape


def test_catalog_limits_match_the_spec_table() -> None:
    assert OPENAI_IMAGE_JSON.max_references == 1
    assert SEEDANCE_VIDEO.max_references == 2
    assert DASHSCOPE_IMAGE_BACKGROUND.max_references == 2
    assert DASHSCOPE_VIDEO.max_references == 1
    assert QWEN_IMAGE_MESSAGES.max_references is None
    assert DASHSCOPE_IMAGE_T2I.max_references == 0
    assert MINIMAX_TTS.max_references == 0
    assert MINIMAX_IMAGE.max_references == 0
    assert GENERIC_JSON.max_references == 0
    assert OPENAI_ASR.accepted == frozenset({MediaType.AUDIO})
    assert OPENAI_ASR.max_references == 1
