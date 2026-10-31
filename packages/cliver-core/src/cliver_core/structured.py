"""Structured output — schema instruction, protocol wire mapping, validation.

Protocol-level only: the OpenAI protocol takes ``response_format`` and the
Anthropic protocol takes ``output_config``; no brand entries, no vendor
branches.  A schema instruction always rides the single system message, so
``mode="prompt"`` is fully functional and the schema is never sent twice.

Parsing accepts a bare JSON object or one wrapped in a single Markdown
fence.  Pydantic schemas validate through ``model_validate``; plain
JSON-Schema dicts are parsed and must yield an object (no JSON-Schema
validator dependency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from cliver_core.messages import UsageInfo
from cliver_core.provider import CLIverResponse

#: ``auto`` — the protocol's broadest option; ``json_schema`` — the strict
#: schema wire option where the protocol has one; ``json_object`` — JSON mode
#: only (prompt fallback on Anthropic); ``prompt`` — no wire option at all.
StructuredMode = Literal["auto", "json_schema", "json_object", "prompt"]

STRUCTURED_MODES: tuple[str, ...] = ("auto", "json_schema", "json_object", "prompt")

_INSTRUCTION = (
    "Respond with a single JSON object conforming to the JSON Schema below. "
    "Return the JSON object only — no prose, no explanations, no Markdown fences.\n"
    "JSON Schema:\n{schema}"
)


@dataclass
class StructuredResponse:
    """A validated structured answer plus its accounting.

    ``value`` is a pydantic model instance (schema given as a model class) or
    the parsed ``dict`` (schema given as a JSON-Schema dict).  ``response`` is
    the final :class:`CLIverResponse`; ``usage_total`` sums every attempt —
    failed ones included; ``attempts`` counts them; ``run_id`` is shared by
    every attempt's events.
    """

    value: BaseModel | dict[str, Any]
    response: CLIverResponse
    attempts: int
    usage_total: UsageInfo | None
    run_id: str


class StructuredOutputError(RuntimeError):
    """Every attempt failed to produce output matching the schema.

    Carries the diagnostics an application needs to log or re-prompt:
    ``attempts``, the last validation ``last_error``, the model's
    ``raw_content``, the final ``response`` (``None`` only if no attempt
    completed), and the ``run_id``.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: str,
        raw_content: str | None,
        run_id: str,
        response: CLIverResponse | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.raw_content = raw_content
        self.run_id = run_id
        self.response = response


def is_pydantic_schema(schema: Any) -> bool:
    """True when *schema* is a pydantic model class (not an instance)."""
    return isinstance(schema, type) and issubclass(schema, BaseModel)


def schema_dict(schema: Any) -> dict[str, Any]:
    """The JSON Schema for *schema* — pydantic models or a dict, nothing else."""
    if is_pydantic_schema(schema):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    raise TypeError(f"schema must be a pydantic BaseModel subclass or a JSON Schema dict, got {type(schema).__name__}")


def schema_name(schema: Any) -> str:
    """Wire name for the schema — the model class name or a generic default."""
    if is_pydantic_schema(schema):
        return schema.__name__
    return "structured_response"


def schema_instruction(schema: Any) -> str:
    """The system-prompt part: answer with one JSON object matching the schema."""
    compact = json.dumps(schema_dict(schema), separators=(",", ":"), ensure_ascii=False)
    return _INSTRUCTION.format(schema=compact)


def protocol_options(protocol: str, *, schema: Any, mode: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Request options carrying the mode's wire mapping — caller keys always win.

    OpenAI-protocol endpoints: ``auto`` sends ``{"type": "json_object"}``
    (broad compatibility; the schema instruction still carries the shape) and
    ``json_schema`` sends the strict schema.  Anthropic-protocol endpoints:
    ``auto``/``json_schema`` send ``output_config.format`` and
    ``json_object`` degrades to prompt mode (no wire option).  The protocol is
    read from the primary ``llm`` binding; callers whose fallbacks speak the
    other protocol should pass ``mode="prompt"`` or explicit options.
    """
    merged = dict(options or {})
    if mode == "prompt":
        return merged
    shape = schema_dict(schema)
    if protocol == "anthropic":
        if mode == "json_object":
            return merged
        if "output_config" not in merged:
            merged["output_config"] = {"format": {"type": "json_schema", "schema": shape}}
        return merged
    if "response_format" in merged:
        return merged
    if mode == "json_schema":
        merged["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name(schema), "schema": shape, "strict": True},
        }
    else:  # auto
        merged["response_format"] = {"type": "json_object"}
    return merged


def _strip_fence(text: str) -> str:
    """Unwrap one Markdown ``` fence around the whole payload."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    newline = stripped.find("\n")
    if newline == -1:
        return stripped
    body = stripped[newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def parse_structured(content: str | None, schema: Any) -> BaseModel | dict[str, Any]:
    """Parse and validate one assistant answer against *schema*.

    Raises :class:`ValueError` (including ``json.JSONDecodeError`` and
    pydantic ``ValidationError``) for empty content, invalid JSON, a
    non-object dict payload, or a pydantic validation failure.
    """
    if content is None or not content.strip():
        raise ValueError("the model returned no content")
    data = json.loads(_strip_fence(content))
    if is_pydantic_schema(schema):
        return schema.model_validate(data)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def corrective_message(error: str) -> str:
    """The user turn that asks for a corrected answer after a failed attempt."""
    return (
        "Your previous response did not conform to the required JSON schema. "
        f"Error: {error}\n"
        "Return only the JSON object that conforms to the schema."
    )
