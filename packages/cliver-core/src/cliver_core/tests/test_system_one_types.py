"""System One typed questions and wire parsing.

The wire shape is shared by every backend (Jev, Bailian, local laya), so it
lives in one module and is tested against the payload observed from the live
Bailian decision-model endpoint.
"""

from __future__ import annotations

import pytest

from cliver_core.errors import ProviderError
from cliver_core.system_one import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResult,
    build_request,
    parse_result,
)


def test_choice_serializes_with_label_descriptions() -> None:
    question = Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "money and invoices", "technical": "bugs and outages"},
    )
    assert question.to_wire() == {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "money and invoices", "technical": "bugs and outages"},
    }


def test_score_serializes_levels_as_a_list() -> None:
    question = Score(instructions="How urgent is this?", criteria=["can wait", "today"])
    assert question.to_wire() == {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "today"],
    }


def test_noul_serializes_without_criteria() -> None:
    question = Noul(instructions="Is this message about billing?")
    assert question.to_wire() == {
        "type": "noul",
        "instructions": "Is this message about billing?",
    }


def test_questions_reject_empty_instructions() -> None:
    with pytest.raises(ValueError, match="instructions"):
        Noul(instructions="  ")


def test_choice_rejects_empty_criteria() -> None:
    with pytest.raises(ValueError, match="criteria"):
        Choice(instructions="Pick one", criteria={})


def test_score_requires_at_least_two_levels() -> None:
    with pytest.raises(ValueError, match="criteria"):
        Score(instructions="How urgent?", criteria=["today"])


def test_build_request_serializes_state_and_questions() -> None:
    body = build_request(
        model="decision-model-preview",
        state={"message": "charged twice"},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "team": Choice(instructions="Which team?", criteria={"a": "x", "b": "y"}),
        },
    )
    assert body["model"] == "decision-model-preview"
    assert body["state"] == {"message": "charged twice"}
    assert body["questions"]["billing"] == {
        "type": "noul",
        "instructions": "Is this about billing?",
    }
    assert body["questions"]["team"]["type"] == "choice"


def test_build_request_omits_model_when_none() -> None:
    body = build_request(
        model=None,
        state="text",
        questions={"q": Noul(instructions="Yes?")},
    )
    assert "model" not in body


def test_parse_result_reads_choice_noul_and_score() -> None:
    payload = {
        "model": "decision-model-preview",
        "request_id": "req-1",
        "answers": {
            "billing": {"type": "noul", "noul": 0.98},
            "department": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.92,
                "probabilities": {"billing": 0.94, "technical": 0.01, "sales": 0.05},
            },
            "urgency": {
                "type": "score",
                "score": 1.15,
                "legend": {"0": "can wait", "1": "this week", "2": "today"},
                "probabilities": {"0": 0.13, "1": 0.59, "2": 0.28},
                "confidence": 0.79,
            },
        },
        "usage": {"input_tokens": 79},
        "latency_ms": 51.8,
    }

    result = parse_result(payload)

    assert isinstance(result, SystemOneResult)
    assert result.model == "decision-model-preview"
    assert result.request_id == "req-1"
    assert result.usage == {"input_tokens": 79}
    assert result.latency_ms == pytest.approx(51.8)

    billing = result.answers["billing"]
    assert isinstance(billing, NoulAnswer)
    assert billing.noul == pytest.approx(0.98)

    department = result.answers["department"]
    assert isinstance(department, ChoiceAnswer)
    assert department.choice == "billing"
    assert department.confidence == pytest.approx(0.92)
    assert department.probabilities["billing"] == pytest.approx(0.94)

    urgency = result.answers["urgency"]
    assert isinstance(urgency, ScoreAnswer)
    assert urgency.score == pytest.approx(1.15)
    assert urgency.legend == {0: "can wait", 1: "this week", 2: "today"}
    assert urgency.probabilities[1] == pytest.approx(0.59)
    assert urgency.confidence == pytest.approx(0.79)


def test_parse_result_keeps_optional_fields_optional() -> None:
    result = parse_result({"answers": {"department": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}}}})
    department = result.answers["department"]
    assert isinstance(department, ChoiceAnswer)
    assert department.confidence is None
    assert result.model is None
    assert result.usage is None


def test_parse_result_rejects_unknown_answer_type() -> None:
    with pytest.raises(ProviderError, match="billing"):
        parse_result({"answers": {"billing": {"type": "mystery"}}})


def test_parse_result_rejects_missing_choice() -> None:
    with pytest.raises(ProviderError, match="department"):
        parse_result({"answers": {"department": {"type": "choice", "probabilities": {}}}})


def test_parse_result_rejects_missing_noul() -> None:
    with pytest.raises(ProviderError, match="billing"):
        parse_result({"answers": {"billing": {"type": "noul"}}})


def test_parse_result_rejects_missing_score() -> None:
    with pytest.raises(ProviderError, match="urgency"):
        parse_result({"answers": {"urgency": {"type": "score"}}})


def test_parse_result_rejects_missing_answers() -> None:
    with pytest.raises(ProviderError, match="answers"):
        parse_result({"model": "decision-model-preview"})
