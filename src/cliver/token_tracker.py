"""
Token Tracker — tracks LLM token consumption with audit logging.

Records input/output tokens per LLM call to monthly JSONL audit logs.
Provides in-memory session totals and audit log queries with filtering.

Audit log location: {config_dir}/audit_logs/{YYYY-MM}.jsonl
Each line: {"ts":"...","model":"...","agent":"...","in":N,"out":N}
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages.base import BaseMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token counts for a single LLM call or aggregation."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        return self

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_tokens(n: int) -> str:
    """Format a token count as a human-readable string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Token extraction from LLM responses
# ---------------------------------------------------------------------------


def extract_usage(response: BaseMessage) -> TokenUsage:
    """Extract token usage from a langchain LLM response.

    Tries structured usage_metadata first, then falls back to
    response_metadata dict (OpenAI-style token_usage).
    Extracts cached token counts from provider-specific fields.
    """
    # Try structured usage_metadata (newer langchain / some providers)
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta:
        if isinstance(usage_meta, dict):
            input_tok = usage_meta.get("input_tokens", 0) or 0
            output_tok = usage_meta.get("output_tokens", 0) or 0
        else:
            input_tok = getattr(usage_meta, "input_tokens", 0) or 0
            output_tok = getattr(usage_meta, "output_tokens", 0) or 0
        if input_tok or output_tok:
            cached = _extract_cached_tokens(response)
            return TokenUsage(input_tokens=input_tok, output_tokens=output_tok, cached_tokens=cached)

    # Fallback: response_metadata dict (OpenAI-style)
    resp_meta = getattr(response, "response_metadata", None) or {}
    token_usage = resp_meta.get("token_usage", {})
    if token_usage:
        cached = _extract_cached_tokens(response)
        return TokenUsage(
            input_tokens=token_usage.get("prompt_tokens", 0) or 0,
            output_tokens=token_usage.get("completion_tokens", 0) or 0,
            cached_tokens=cached,
        )

    return TokenUsage()


def _extract_cached_tokens(response: BaseMessage) -> int:
    """Extract cached token count from provider-specific response fields.

    Supports:
    - OpenAI/GLM: usage.prompt_tokens_details.cached_tokens
    - DeepSeek: usage.prompt_cache_hit_tokens
    - LangChain usage_metadata: input_token_details.cache_read
    """
    # LangChain usage_metadata (newer versions)
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta:
        if isinstance(usage_meta, dict):
            details = usage_meta.get("input_token_details", {})
            if isinstance(details, dict):
                cached = details.get("cache_read", 0)
                if cached:
                    return cached
        else:
            details = getattr(usage_meta, "input_token_details", None)
            if details:
                cached = getattr(details, "cache_read", 0)
                if cached:
                    return cached

    # OpenAI/GLM: response_metadata.token_usage.prompt_tokens_details.cached_tokens
    resp_meta = getattr(response, "response_metadata", None) or {}
    token_usage = resp_meta.get("token_usage", {})
    prompt_details = token_usage.get("prompt_tokens_details", {})
    if isinstance(prompt_details, dict):
        cached = prompt_details.get("cached_tokens", 0)
        if cached:
            return cached

    # DeepSeek: response_metadata.token_usage.prompt_cache_hit_tokens
    cached = token_usage.get("prompt_cache_hit_tokens", 0)
    if cached:
        return cached

    return 0


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------


class TokenTracker:
    """Tracks token consumption with in-memory session totals and audit logging.

    Audit logs: {audit_dir}/{YYYY-MM}.jsonl — monthly rotation, one JSON per line.
    """

    def __init__(self, audit_dir: Path, agent_name: str = "CLIver"):
        self.audit_dir = audit_dir
        self.agent_name = agent_name
        # In-memory session totals: model_name → TokenUsage
        self.session_totals: Dict[str, TokenUsage] = {}
        # Last recorded usage (for per-query display)
        self.last_usage: Optional[TokenUsage] = None
        self.last_model: Optional[str] = None

    def record(self, model: str, usage: TokenUsage) -> None:
        """Record a token usage event — update session totals + append to audit log."""
        if usage.total_tokens == 0:
            return

        # Update in-memory session totals
        if model not in self.session_totals:
            self.session_totals[model] = TokenUsage()
        self.session_totals[model] += usage
        self.last_usage = usage
        self.last_model = model

        # Append to audit log
        self._append_audit(model, usage)

    def get_session_summary(self) -> Dict[str, TokenUsage]:
        """Get in-memory session totals by model."""
        return dict(self.session_totals)

    def get_session_total(self) -> TokenUsage:
        """Get aggregate session total across all models."""
        total = TokenUsage()
        for usage in self.session_totals.values():
            total += usage
        return total

    def query(
        self,
        model: Optional[str] = None,
        agent: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, TokenUsage]]:
        """Query audit logs with optional filters.

        Returns: {model_name: {agent_name: TokenUsage}}
        """
        results: Dict[str, Dict[str, TokenUsage]] = {}

        for record in self._read_audit_logs(start, end):
            rec_model = record.get("model", "unknown")
            rec_agent = record.get("agent", "unknown")

            # Apply filters
            if model and rec_model != model:
                continue
            if agent and rec_agent != agent:
                continue

            if rec_model not in results:
                results[rec_model] = {}
            if rec_agent not in results[rec_model]:
                results[rec_model][rec_agent] = TokenUsage()

            results[rec_model][rec_agent] += TokenUsage(
                input_tokens=record.get("in", 0),
                output_tokens=record.get("out", 0),
                cached_tokens=record.get("cached", 0),
            )

        return results

    # -- Audit log I/O ---------------------------------------------------------

    def _append_audit(self, model: str, usage: TokenUsage) -> None:
        """Append a record to the current month's audit log."""
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        filename = now.strftime("%Y-%m") + ".jsonl"
        filepath = self.audit_dir / filename

        record = {
            "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "agent": self.agent_name,
            "in": usage.input_tokens,
            "out": usage.output_tokens,
        }
        if usage.cached_tokens > 0:
            record["cached"] = usage.cached_tokens

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_audit_logs(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Read and filter audit log records."""
        records = []
        if not self.audit_dir.is_dir():
            return records

        for path in sorted(self.audit_dir.glob("*.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Time filter
                    ts_str = record.get("ts", "")
                    if ts_str and (start or end):
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if start and ts < start:
                                continue
                            if end and ts > end:
                                continue
                        except ValueError:
                            pass

                    records.append(record)

        return records
