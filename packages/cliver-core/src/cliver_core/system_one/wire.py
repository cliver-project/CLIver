"""JSON wire conversion for the System One request/response shape.

Jev (TypeSafe), Aliyun Bailian's decision model, and a local ``laya-serve``
all speak the same HTTP payload::

    POST .../v1/systemone
    {"model": "...", "state": ..., "questions": {"name": {...}}}
    -> {"model": "...", "request_id": "...", "answers": {...}, "usage": {...}}

Only the answers are normalized here.  Anything malformed is a
:class:`ProviderError` naming the offending question — never a silent
empty answer.
"""

from __future__ import annotations

from typing import Any, Mapping

from cliver_core.errors import ProviderError
from cliver_core.system_one.types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneResult,
)

__all__ = ["build_request", "parse_answers", "parse_result"]


def build_request(
    *,
    model: str | None,
    state: Any,
    questions: Mapping[str, Question],
) -> dict[str, Any]:
    """Build the POST body shared by every backend."""

    body: dict[str, Any] = {
        "state": state,
        "questions": {name: question.to_wire() for name, question in questions.items()},
    }
    if model:
        body["model"] = model
    return body


def _number(value: Any, *, question: str, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"System One answer {question!r} has a non-numeric {field}: {value!r}") from exc


def _optional_number(value: Any, *, question: str, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, question=question, field=field)


def _choice_answer(name: str, raw: Mapping[str, Any]) -> ChoiceAnswer:
    choice = raw.get("choice")
    if not isinstance(choice, str) or not choice:
        raise ProviderError(f"System One answer {name!r} is a choice without a 'choice'")
    probabilities = raw.get("probabilities") or {}
    if not isinstance(probabilities, Mapping):
        raise ProviderError(f"System One answer {name!r} has malformed probabilities")
    return ChoiceAnswer(
        choice=choice,
        probabilities={
            str(label): _number(value, question=name, field=f"probability[{label!r}]")
            for label, value in probabilities.items()
        },
        confidence=_optional_number(raw.get("confidence"), question=name, field="confidence"),
        answer_confidence=_optional_number(raw.get("answer_confidence"), question=name, field="answer_confidence"),
    )


def _score_answer(name: str, raw: Mapping[str, Any]) -> ScoreAnswer:
    if raw.get("score") is None:
        raise ProviderError(f"System One answer {name!r} is a score without a 'score'")
    probabilities = raw.get("probabilities") or {}
    legend = raw.get("legend") or {}
    if not isinstance(probabilities, Mapping) or not isinstance(legend, Mapping):
        raise ProviderError(f"System One answer {name!r} has malformed score data")
    return ScoreAnswer(
        score=_number(raw["score"], question=name, field="score"),
        probabilities={
            int(level): _number(value, question=name, field=f"probability[{level}]")
            for level, value in probabilities.items()
        },
        legend={int(level): str(label) for level, label in legend.items()},
        confidence=_optional_number(raw.get("confidence"), question=name, field="confidence"),
    )


def _noul_answer(name: str, raw: Mapping[str, Any]) -> NoulAnswer:
    if raw.get("noul") is None:
        raise ProviderError(f"System One answer {name!r} is a noul without a 'noul'")
    return NoulAnswer(noul=_number(raw["noul"], question=name, field="noul"))


def _parse_answer(name: str, raw: Any) -> Answer:
    if not isinstance(raw, Mapping):
        raise ProviderError(f"System One answer {name!r} is not an object")
    kind = raw.get("type")
    if kind == "choice":
        return _choice_answer(name, raw)
    if kind == "score":
        return _score_answer(name, raw)
    if kind == "noul":
        return _noul_answer(name, raw)
    raise ProviderError(f"System One answer {name!r} has an unknown type: {kind!r}")


def parse_answers(raw_answers: Mapping[str, Any]) -> dict[str, Answer]:
    """Normalize the ``answers`` mapping of a System One response."""

    return {name: _parse_answer(name, raw) for name, raw in raw_answers.items()}


def parse_result(payload: Mapping[str, Any]) -> SystemOneResult:
    """Normalize a full System One response body."""

    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise ProviderError("System One response is missing an 'answers' object")
    usage = payload.get("usage")
    latency = payload.get("latency_ms")
    return SystemOneResult(
        answers=parse_answers(raw_answers),
        model=payload.get("model"),
        request_id=payload.get("request_id"),
        usage=dict(usage) if isinstance(usage, Mapping) else None,
        latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
    )
