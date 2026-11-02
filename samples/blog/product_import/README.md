# A product URL in, validated JSON out

`chat_structured()` runs the normal agent loop and validates the final answer against a pydantic
schema. A parse or validation failure is re-prompted with the error; the caller either gets a
`Product` instance or a `StructuredOutputError` with the raw answer and attempt count.

Fetching the page is ordinary application code (for example `httpx.get(url).text`); this sample
starts from the extracted text so its test needs no network.

## Install and run

```bash
pip install cliver-core==0.3.0
export DEEPSEEK_API_KEY=...   # variable name only; never commit a value
python product_import.py
```

From this repository (workspace install):

```bash
uv run python samples/blog/product_import/product_import.py
```

Expected output: indented JSON for the product, then `attempts: 1`.

## Test (no key, no network)

From `samples/`:

```bash
uv run pytest blog/product_import -q
```

One test returns valid JSON on the first attempt; the other returns prose first, proving the
corrective retry reaches the model and the second answer validates.
