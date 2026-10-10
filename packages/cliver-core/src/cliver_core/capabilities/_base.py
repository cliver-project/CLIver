"""Shared plumbing for the standalone capability calls.

Every capability in this package is a self-contained async function: given a
:class:`~cliver_core.bindings.ModelBinding` — the one handle carrying the
model definition, its provider, the endpoint ``path`` override, and the
default call parameters — plus purpose-specific parameters, it performs the
call and returns the result.  No AgentCore required.

The binding is the whole handle: the provider comes from
:meth:`~cliver_core.bindings.ModelBinding.get_provider`, the model name from
``binding.model``, and the default request parameters from
:meth:`~cliver_core.bindings.ModelBinding.call_defaults`.  This module only
holds the pieces every capability shares:

* :func:`coerce_media` — normalize bytes / path / URL / ``MediaContent``
  into :class:`~cliver_core.media.MediaContent`.
* :func:`drop_none` (from :mod:`cliver_core.util`) — build request parameter
  dicts that omit unset values.
"""

from __future__ import annotations

import base64
from pathlib import Path

from cliver_core.media import MediaContent, MediaType, load_media_file
from cliver_core.util import drop_none

__all__ = ["coerce_media", "drop_none"]


def coerce_media(source: MediaContent | bytes | str | Path) -> MediaContent:
    """Normalize *source* into :class:`MediaContent`.

    ``MediaContent`` passes through untouched, raw ``bytes`` are wrapped as WAV
    audio, and anything else (path or URL, ``str`` or ``Path``) is loaded via
    :func:`~cliver_core.media.load_media_file`.
    """
    if isinstance(source, MediaContent):
        return source
    if isinstance(source, (bytes, bytearray)):
        return MediaContent(
            type=MediaType.AUDIO,
            data=base64.b64encode(bytes(source)).decode("utf-8"),
            mime_type="audio/wav",
        )
    return load_media_file(str(source))
