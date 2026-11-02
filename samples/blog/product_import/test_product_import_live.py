"""Live check — needs CLIVER_LIVE_VERIFICATION=1 and DEEPSEEK_API_KEY."""

from __future__ import annotations

from _shared.testing import skip_unless_live
from product_import import PAGE_TEXT, Product, extract_product


async def test_product_import_live() -> None:
    skip_unless_live("DEEPSEEK_API_KEY")

    result = await extract_product(PAGE_TEXT)

    assert isinstance(result.value, Product)
    assert result.value.currency == "USD"
