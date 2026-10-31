"""Video generation — standalone capability call.

Works out of the box with just a :class:`~cliver_core.bindings.ModelBinding`::

    from cliver_core.bindings import ModelBinding
    from cliver_core.capabilities import generate_video

    response = await generate_video(
        prompt="a drone shot over a misty forest",
        binding=ModelBinding.video(
            model="video-01",
            api_key="sk-...",
            base_url="https://api.example.com/v1",
        ),
        output_dir="./out",
    )

This is the implementation AgentCore's ``agent.video(...)`` routes to, passing
its ``bindings.video`` binding plus any call-site parameters.

Long-running: many vendors take minutes to finish, so this call may block for
a while — AgentCore's ``submit_video(...)`` wraps it in a background
:class:`~cliver_core.media_jobs.MediaJob`, which is the right shape for
interactive use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cliver_core.capabilities._base import drop_none
from cliver_core.media import MediaContent
from cliver_core.provider import CLIverResponse

if TYPE_CHECKING:
    from cliver_core.bindings import ModelBinding


async def generate_video(
    *,
    prompt: str,
    binding: ModelBinding,
    media: list[MediaContent] | None = None,
    output_dir: str | None = None,
    duration: int | None = None,
    resolution: str | None = None,
    **options: Any,
) -> CLIverResponse:
    """Generate a video from *prompt* with the model in *binding*.

    Args:
        prompt: The video prompt.
        binding: The :class:`~cliver_core.bindings.ModelBinding` to run on
            (AgentCore passes its ``bindings.video``) — it carries the model
            definition, its cached provider, the endpoint ``path`` override,
            and the default parameters.  Model configuration is never passed
            as loose arguments.
        media: Reference media for image-to-video etc.
        output_dir: Save generated files here, if given.
        duration: Clip duration in seconds.
        resolution: Video resolution (e.g. ``"1080p"``).
        **options: Extra generation parameters, forwarded verbatim — a
            call-site ``path`` overrides the binding's endpoint path.

    Returns:
        A :class:`~cliver_core.provider.CLIverResponse`; generated videos are
        in ``.media``.
    """
    async with binding.lease_provider() as provider:
        return await provider.video(
            prompt,
            binding.model,
            media=media,
            output_dir=output_dir,
            **{
                **binding.call_defaults(),
                **drop_none(duration=duration, resolution=resolution),
                **options,
            },
        )
