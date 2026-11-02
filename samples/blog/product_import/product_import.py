"""Turn a product page into a validated record with ``chat_structured()``.

Run it with ``DEEPSEEK_API_KEY`` set:

    python samples/blog/product_import/product_import.py
"""

from __future__ import annotations

import asyncio

from cliver_core import AgentCore, ModelBindings, StructuredResponse, presets
from pydantic import BaseModel, Field

# region: schema
class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Numeric price, no currency symbol")
    currency: str = Field(description="ISO 4217 currency code")
    in_stock: bool


# endregion: schema


# region: extract
PAGE_TEXT = """Acme Travel Mug — $18.00 USD. Double-walled steel, 350 ml.
In stock. Ships within two business days."""


async def extract_product(page_text: str, agent: AgentCore | None = None) -> StructuredResponse:
    owned = agent is None
    agent = agent or AgentCore(ModelBindings(llm=presets.deepseek()))
    try:
        return await agent.chat_structured(
            f"Extract the product from this page:\n\n{page_text}",
            schema=Product,
            mode="auto",
        )
    finally:
        if owned:
            await agent.aclose()


# endregion: extract


async def main() -> None:
    result = await extract_product(PAGE_TEXT)
    print(result.value.model_dump_json(indent=2))
    print(f"attempts: {result.attempts}")


if __name__ == "__main__":
    asyncio.run(main())
