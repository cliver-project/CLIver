"""The structured-extraction sample runs offline and covers the retry path."""

from __future__ import annotations

from _shared.testing import ScriptedProvider, build_scripted_agent
from product_import import PAGE_TEXT, Product, extract_product


async def test_extract_product_returns_a_validated_model() -> None:
    provider = ScriptedProvider(['{"name": "Acme Travel Mug", "price": 18.0, "currency": "USD", "in_stock": true}'])
    agent = build_scripted_agent(provider)

    result = await extract_product(PAGE_TEXT, agent)

    assert isinstance(result.value, Product)
    assert result.value.price == 18.0
    assert result.attempts == 1


async def test_extract_product_retries_after_invalid_json() -> None:
    provider = ScriptedProvider(
        [
            "I found the product: Acme Travel Mug, $18.",
            '{"name": "Acme Travel Mug", "price": 18.0, "currency": "USD", "in_stock": true}',
        ]
    )
    agent = build_scripted_agent(provider)

    result = await extract_product(PAGE_TEXT, agent)

    assert result.attempts == 2
    assert result.value.currency == "USD"
    assert len(provider.requests) == 2
    assert "did not conform" in provider.requests[1].messages[-1].content
