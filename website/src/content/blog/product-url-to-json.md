---
title: "A product URL in, validated JSON out"
description: "chat_structured() asks for JSON, validates it, and retries with the error — so your code gets a typed object or a clear failure."
pubDate: 2026-11-11
tags:
  - Structured Output
  - Pydantic
issue: 3
draft: false
author: Lin Gao
series: Field Notes
---

Asking a model for JSON works until it returns prose, a Markdown fence, or a number where you
expected a string. Then you write a parser, a retry prompt, and a second parser.

`chat_structured()` puts that contract in one call: the schema goes into the system message, the
provider gets its native structured-output option when it has one, and the answer is validated
before it reaches your code.

## The shape you want

```python title="product_import.py" file="samples/blog/product_import/product_import.py" region="schema"
```

Pydantic gives you the validation, the field descriptions, and the JSON Schema in one place. A
JSON-Schema `dict` works too and returns a plain `dict`.

## The call

```python title="product_import.py" file="samples/blog/product_import/product_import.py" region="extract"
```

The sample starts from page text so its test needs no network; fetching the URL is ordinary
application code (`httpx.get(url).text`) before this call.

On success, `result.value` is a `Product`. On a parse or validation failure, the core appends the
bad answer and a corrective user turn, then asks again — up to `retries` extra attempts. The
caller's `conversation` list is never mutated, and `usage_total` sums every attempt, failed ones
included.

If every attempt fails, `StructuredOutputError` carries `attempts`, `last_error`, `raw_content`,
`run_id`, and the final response. That is enough to log the failure or route it to a human.

## The run

```text
{
  "name": "Acme Travel Mug",
  "price": 18.0,
  "currency": "USD",
  "in_stock": true
}
attempts: 1
```

The first answer validated. A prose answer would show `attempts: 2`, and the second request would
carry the parse error back to the model.

## Caveats

- The final answer must be assistant text; streaming structured output is not supported.
- `mode="auto"` picks the broadest wire option for the binding's protocol. `json_object` on the
  Anthropic protocol cannot enforce a schema, so it behaves like `prompt`.
- Any `response_format` or `output_config` you pass in `options` wins over the mode mapping.

The CLI runs one task per invocation or a streaming REPL; `chat_structured()` is a library API
today, so this pattern lives in your service, not the terminal. The CLI is experimental and not
on PyPI yet.

Next: [Structured Output](../../core-concepts/structured-output/) covers modes, retries, and
provider caveats in full.
