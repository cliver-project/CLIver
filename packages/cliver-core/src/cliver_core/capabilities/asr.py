"""Speech-to-text (ASR) — standalone capability call.

Works out of the box with just a :class:`~cliver_core.bindings.ModelBinding`::

    from cliver_core.bindings import ModelBinding
    from cliver_core.capabilities import transcribe_audio

    response = await transcribe_audio(
        audio="voice-note.wav",          # path, URL, bytes, or MediaContent
        binding=ModelBinding.asr(
            model="whisper-1",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            language="en",
        ),
    )
    print(response.message.content)      # the transcript

This is the implementation AgentCore's ``agent.asr(...)`` routes to, passing
its ``bindings.asr`` binding plus any call-site parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cliver_core.capabilities._base import coerce_media, drop_none
from cliver_core.media import MediaContent
from cliver_core.provider import CLIverResponse

if TYPE_CHECKING:
    from cliver_core.bindings import ModelBinding


async def transcribe_audio(
    *,
    audio: MediaContent | bytes | str | Path,
    binding: ModelBinding,
    language: str | None = None,
    **options: Any,
) -> CLIverResponse:
    """Transcribe *audio* with the model in *binding*.

    Args:
        audio: The audio to transcribe — a
            :class:`~cliver_core.media.MediaContent`, raw ``bytes`` (assumed
            WAV), or a file path / URL.
        binding: The :class:`~cliver_core.bindings.ModelBinding` to run on
            (AgentCore passes its ``bindings.asr``) — it carries the model
            definition, its cached provider, the endpoint ``path`` override,
            and the default parameters.  Model configuration is never passed
            as loose arguments.
        language: Optional language hint for the transcription — a BCP-47
            tag (``en``, ``zh``, …).  Sent as a multipart form field in the
            OpenAI fashion, or an HTTP header on providers that take it
            there (MiniMax); omit it on those providers for mixed-language
            recognition.
        **options: Extra transcription parameters, forwarded verbatim — a
            call-site ``path`` overrides the binding's endpoint path.
            ``response_format`` selects the transcript shape: ``json`` /
            ``verbose_json`` put the transcript in ``.message.content``,
            while ``srt`` / ``vtt`` deliver timed subtitles as plain text.
            ``stream=True`` raises ``ValueError`` — streaming transcription
            is not implemented.

    Returns:
        A :class:`~cliver_core.provider.CLIverResponse`; the transcript is in
        ``.message.content``.  A provider that does not implement
        transcription raises
        :class:`~cliver_core.errors.CapabilityNotSupported`.
    """
    async with binding.lease_provider() as provider:
        return await provider.asr(
            coerce_media(audio),
            binding.model,
            **{
                **binding.call_defaults(),
                **drop_none(language=language),
                **options,
            },
        )
