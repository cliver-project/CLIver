"""System One decision models: typed predictions instead of generated text.

A System One model reads a ``state`` and a set of typed questions and returns
structured answers — a selected ``choice``, a probability-weighted ``score``,
or ``noul`` (P(true)).  The same wire shape is served by Jev (TypeSafe), the
Aliyun Bailian decision model, and a local ``laya-serve``; a local ``laya``
backend implements it in process.

The selector built on top of this contract lives in
:mod:`cliver_core.system_one.selector`: it ranks tools and skills by their
probability of fitting the user's request so that large catalogues are
narrowed before they reach the main model.
"""

from __future__ import annotations

from cliver_core.system_one.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResult,
)
from cliver_core.system_one.wire import build_request, parse_answers, parse_result

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
    "build_request",
    "parse_answers",
    "parse_result",
]
