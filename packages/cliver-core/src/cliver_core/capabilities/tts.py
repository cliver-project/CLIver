"""Text-to-speech — standalone capability call.

Works out of the box with just a :class:`~cliver_core.bindings.ModelBinding`::

    from cliver_core.bindings import ModelBinding
    from cliver_core.capabilities import text_to_speech

    response = await text_to_speech(
        text="Hello from CLIver.",
        binding=ModelBinding.tts(
            model="gpt-4o-mini-tts",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            voice="alloy",
        ),
        response_format="mp3",
        output_dir="./out",
    )

This is the implementation AgentCore's ``agent.tts(...)`` (and its legacy
``agent.audio(...)`` alias) routes to, passing its ``bindings.tts`` /
``bindings.audio`` binding plus any call-site parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cliver_core.capabilities._base import drop_none
from cliver_core.provider import CLIverResponse

if TYPE_CHECKING:
    from cliver_core.bindings import ModelBinding


async def text_to_speech(
    *,
    text: str,
    binding: ModelBinding,
    voice: str | None = None,
    response_format: str | None = None,
    output_dir: str | None = None,
    **options: Any,
) -> CLIverResponse:
    """Synthesize speech from *text* with the model in *binding*.

    Args:
        text: The text to speak.
        binding: The :class:`~cliver_core.bindings.ModelBinding` to run on
            (AgentCore passes its ``bindings.tts`` / ``bindings.audio``) — it
            carries the model definition, its cached provider, the endpoint
            ``path`` override, and the default parameters.  Model
            configuration is never passed as loose arguments.
        voice: Voice name (vendor-specific; e.g. ``"alloy"``).
        response_format: Audio format (e.g. ``"mp3"``, ``"wav"``).
        output_dir: Save the audio file here, if given.
        **options: Extra generation parameters, forwarded verbatim — a
            call-site ``path`` overrides the binding's endpoint path.

    Returns:
        A :class:`~cliver_core.provider.CLIverResponse`; the generated audio is
        in ``.media``.
    """
    async with binding.lease_provider() as provider:
        return await provider.tts(
            text,
            binding.model,
            output_dir=output_dir,
            **{
                **binding.call_defaults(),
                **drop_none(voice=voice, response_format=response_format),
                **options,
            },
        )
