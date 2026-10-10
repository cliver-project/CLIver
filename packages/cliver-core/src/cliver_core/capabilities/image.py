"""Image generation — standalone capability call.

Works out of the box with just a :class:`~cliver_core.bindings.ModelBinding`::

    from cliver_core.bindings import ModelBinding
    from cliver_core.capabilities import generate_image

    response = await generate_image(
        prompt="a red panda on a skateboard",
        binding=ModelBinding.image(
            model="gpt-image-1",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            size="1024x1024",
        ),
        n=2,
    )
    for item in response.media or []:
        print(item.saved_path or item.data[:64])

This is the implementation AgentCore's ``agent.image(...)`` routes to, passing
its ``bindings.image`` binding plus any call-site parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cliver_core.capabilities._base import drop_none
from cliver_core.media import MediaContent
from cliver_core.provider import CLIverResponse

if TYPE_CHECKING:
    from cliver_core.bindings import ModelBinding


async def generate_image(
    *,
    prompt: str,
    binding: ModelBinding,
    media: list[MediaContent] | None = None,
    output_dir: str | None = None,
    n: int | None = None,
    size: str | None = None,
    style: str | None = None,
    quality: str | None = None,
    **options: Any,
) -> CLIverResponse:
    """Generate images from *prompt* with the model in *binding*.

    Args:
        prompt: The image prompt.
        binding: The :class:`~cliver_core.bindings.ModelBinding` to run on
            (AgentCore passes its ``bindings.image``) — it carries the model
            definition, its cached provider, the endpoint ``path`` override,
            and the default parameters.  Model configuration is never passed
            as loose arguments.
        media: Reference image(s) for editing / image-to-image.
        output_dir: Save generated files here, if given.
        n: Number of images to generate.
        size: Image size (e.g. ``"1024x1024"``).
        style: Style hint (e.g. ``"vivid"``).
        quality: Quality hint (e.g. ``"high"``).
        **options: Extra generation parameters, forwarded verbatim — a
            call-site ``path`` overrides the binding's endpoint path.

    Returns:
        A :class:`~cliver_core.provider.CLIverResponse`; generated images are
        in ``.media`` (and ``.message.content`` summarizes the call).
    """
    provider = binding.get_provider()
    return await provider.image(
        prompt,
        binding.model,
        media=media,
        output_dir=output_dir,
        **{
            **binding.call_defaults(),
            **drop_none(n=n, size=size, style=style, quality=quality),
            **options,
        },
    )
