"""Text embeddings — standalone capability call.

Works out of the box with just a :class:`~cliver_core.bindings.ModelBinding`::

    from cliver_core.bindings import ModelBinding
    from cliver_core.capabilities import embed

    result = await embed(
        texts=["hello", "world"],
        binding=ModelBinding.embed(
            model="text-embedding-3-small",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        ),
    )
    for vector in result.vectors:
        ...

This is the implementation AgentCore's ``agent.embed(...)`` routes to, passing
its ``bindings.embed`` binding plus any call-site parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cliver_core.provider import EmbeddingsResult

if TYPE_CHECKING:
    from cliver_core.bindings import ModelBinding


async def embed(
    *,
    texts: str | list[str],
    binding: ModelBinding,
    **options: Any,
) -> EmbeddingsResult:
    """Embed *texts* with the model in *binding*.

    Args:
        texts: One string or a list of strings to embed.
        binding: The :class:`~cliver_core.bindings.ModelBinding` to run on
            (AgentCore passes its ``bindings.embed``) — it carries the model
            definition, its cached provider, the endpoint ``path`` override,
            and the default parameters.  Model configuration is never passed
            as loose arguments.
        **options: Extra parameters (e.g. ``dimensions``), forwarded verbatim —
            a call-site ``path`` overrides the binding's endpoint path.

    Returns:
        An :class:`~cliver_core.provider.EmbeddingsResult` with one vector per
        input text, in input order.
    """
    if isinstance(texts, str):
        texts = [texts]
    async with binding.lease_provider() as provider:
        return await provider.embed(
            list(texts),
            model=binding.model,
            **{**binding.call_defaults(), **options},
        )
