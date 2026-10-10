"""Document reranking — standalone capability call.

Works out of the box with just a :class:`~cliver_core.bindings.ModelBinding`::

    from cliver_core.bindings import ModelBinding
    from cliver_core.capabilities import rerank

    items = await rerank(
        query="what is the capital of France?",
        documents=["Paris is the capital of France.", "Berlin is in Germany."],
        binding=ModelBinding.rerank(
            model="rerank-v3.5",
            api_key="...",
            base_url="https://api.example.com/v1",
        ),
        top_k=1,
    )

This is the implementation AgentCore's ``agent.rerank(...)`` routes to, passing
its ``bindings.rerank`` binding plus any call-site parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cliver_core.capabilities._base import drop_none
from cliver_core.provider import RerankItem

if TYPE_CHECKING:
    from cliver_core.bindings import ModelBinding


async def rerank(
    *,
    query: str,
    documents: list[str],
    binding: ModelBinding,
    top_k: int | None = None,
    **options: Any,
) -> list[RerankItem]:
    """Score *documents* against *query* with the model in *binding*.

    Args:
        query: The query to rank against.
        documents: Candidate documents, in any order.
        binding: The :class:`~cliver_core.bindings.ModelBinding` to run on
            (AgentCore passes its ``bindings.rerank``) — it carries the model
            definition, its cached provider, the endpoint ``path`` override,
            and the default parameters.  Model configuration is never passed
            as loose arguments.
        top_k: Return only the top *k* items (also sent to the API).
        **options: Extra parameters, forwarded verbatim — a call-site ``path``
            overrides the binding's endpoint path.

    Returns:
        :class:`~cliver_core.provider.RerankItem` entries, best first.
    """
    provider = binding.get_provider()
    return await provider.rerank(
        query,
        documents,
        model=binding.model,
        **{**binding.call_defaults(), **drop_none(top_k=top_k), **options},
    )
