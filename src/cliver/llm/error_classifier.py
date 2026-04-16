"""
Error classifier for LLM API errors.

Classifies exceptions into three action categories:
- RETRY: Transient errors — retry same model with exponential backoff
- FAILOVER: Permanent errors — switch to next capable model
- FATAL: Unrecoverable errors — stop and return error
"""

import logging
import socket
import ssl
from dataclasses import dataclass
from enum import Enum

from httpx import TimeoutException

logger = logging.getLogger(__name__)

_CONTEXT_OVERFLOW_PATTERNS = [
    "context length",
    "context_length_exceeded",
    "too many tokens",
    "maximum.*token",
    "prompt is too long",
    "input too long",
    "request too large",
]


class ErrorAction(str, Enum):
    RETRY = "retry"
    FAILOVER = "failover"
    FATAL = "fatal"


@dataclass
class ClassifiedError:
    action: ErrorAction
    reason: str
    should_compress: bool
    original_error: Exception


def classify_error(error: Exception) -> ClassifiedError:
    status = _extract_status(error)
    message = str(error).lower()

    if status == 429:
        return ClassifiedError(ErrorAction.RETRY, "rate_limit", False, error)
    if status in (500, 502, 503):
        return ClassifiedError(ErrorAction.RETRY, "server_error", False, error)
    if status in (401, 403):
        return ClassifiedError(ErrorAction.FAILOVER, "auth", False, error)
    if status == 402:
        return ClassifiedError(ErrorAction.FAILOVER, "billing", False, error)
    if status == 404:
        return ClassifiedError(ErrorAction.FAILOVER, "model_not_found", False, error)
    if status == 400:
        if _is_context_overflow(message):
            return ClassifiedError(ErrorAction.FAILOVER, "context_overflow", True, error)
        return ClassifiedError(ErrorAction.FAILOVER, "bad_request", False, error)

    if isinstance(error, TimeoutException):
        return ClassifiedError(ErrorAction.RETRY, "timeout", False, error)
    if isinstance(error, (ConnectionError, socket.gaierror, ssl.SSLError)):
        return ClassifiedError(ErrorAction.RETRY, "connection_error", False, error)

    return ClassifiedError(ErrorAction.RETRY, "unknown", False, error)


def _extract_status(error: Exception) -> int | None:
    if hasattr(error, "status_code"):
        return error.status_code
    response = getattr(error, "response", None)
    if response and hasattr(response, "status_code"):
        return response.status_code
    return None


def _is_context_overflow(message: str) -> bool:
    return any(pattern in message for pattern in _CONTEXT_OVERFLOW_PATTERNS)
