"""Typed questions and answers for System One decision models.

A System One call sends a ``state`` plus named questions and receives one
structured prediction per question — probabilities, never generated text.
These dataclasses are the backend-neutral contract; wire conversion lives in
:mod:`cliver_core.system_one.wire`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "SystemOneResult",
]


@dataclass(frozen=True)
class Choice:
    """Pick one of several labelled options.

    ``criteria`` maps each option label to a one-sentence description.  The
    model scores every option; the highest-probability one becomes the
    answer's ``choice``.
    """

    instructions: str
    criteria: dict[str, str]
    type: ClassVar[str] = "choice"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("Choice instructions must not be empty")
        if not self.criteria:
            raise ValueError("Choice criteria must not be empty")
        if any(not label for label in self.criteria):
            raise ValueError("Choice criteria labels must not be empty")

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Score:
    """Place the state on an ordered scale.

    ``criteria`` lists the levels in ascending order; the answer is their
    probability-weighted expected value together with the full distribution.
    """

    instructions: str
    criteria: list[str]
    type: ClassVar[str] = "score"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("Score instructions must not be empty")
        if len(self.criteria) < 2:
            raise ValueError("Score criteria must contain at least two levels")

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


@dataclass(frozen=True)
class Noul:
    """A single yes/no judgement; the answer is P(true)."""

    instructions: str
    type: ClassVar[str] = "noul"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("Noul instructions must not be empty")

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions}


Question: TypeAlias = Choice | Score | Noul


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None
    answer_confidence: float | None = None
    type: ClassVar[str] = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: dict[int, float]
    legend: dict[int, str]
    confidence: float | None = None
    type: ClassVar[str] = "score"


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: ClassVar[str] = "noul"


Answer: TypeAlias = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(frozen=True)
class SystemOneResult:
    """The parsed answer for every question, plus request metadata."""

    answers: dict[str, Answer]
    model: str | None = None
    request_id: str | None = None
    usage: dict[str, Any] | None = None
    latency_ms: float | None = None
