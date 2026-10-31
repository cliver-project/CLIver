"""SystemOneSelector — System One ranking for large tool/skill catalogues.

The provider is a deterministic in-memory decision model: every ``Choice``
scores labels from a configured table, every ``Noul`` reads the candidate
name quoted in its instructions.  No network.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from cliver_core.bindings import ModelBinding
from cliver_core.errors import ProviderError
from cliver_core.skill import Skill
from cliver_core.system_one import Choice, ChoiceAnswer, Noul, NoulAnswer, SystemOneResult
from cliver_core.system_one.selector import SystemOneSelector
from cliver_core.tool import CLIverTool


class _FakeDecisionModel:
    """A deterministic System One model over a score/fit table."""

    def __init__(self, *, scores: dict[str, float] | None = None, fits: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.fits = fits or {}
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    async def system_one(self, *, state: str, questions: dict, model: str, **options) -> SystemOneResult:
        self.calls.append({"state": state, "questions": questions, "model": model, "options": options})
        if self.error is not None:
            raise self.error
        answers = {}
        for name, question in questions.items():
            if isinstance(question, Choice):
                raw = {label: self.scores.get(label, 0.01) for label in question.criteria}
                total = sum(raw.values())
                probabilities = {label: value / total for label, value in raw.items()}
                best = max(probabilities, key=probabilities.get)
                answers[name] = ChoiceAnswer(
                    choice=best,
                    probabilities=probabilities,
                    confidence=probabilities[best],
                )
            elif isinstance(question, Noul):
                match = re.search(r"'([^']+)'", question.instructions)
                assert match is not None, question.instructions
                answers[name] = NoulAnswer(noul=self.fits.get(match.group(1), 0.5))
            else:  # pragma: no cover - the selector only sends these two
                raise AssertionError(question)
        return SystemOneResult(answers=answers)


def _tool(index: int, *, long_description: str | None = None) -> CLIverTool:
    return CLIverTool(
        name=f"tool_{index:02d}",
        description=f"description for tool {index}",
        long_description=long_description,
        parameters={},
        execute=lambda **_kwargs: [{"text": "ok"}],
    )


def _skill(index: int) -> Skill:
    return Skill(
        name=f"skill_{index:02d}",
        description=f"description for skill {index}",
        body=f"body for skill {index}",
        base_dir=Path("."),
    )


def _selector(model: _FakeDecisionModel, **kwargs: Any) -> SystemOneSelector:
    kwargs.setdefault("min_tools", 20)
    kwargs.setdefault("min_skills", 20)
    return SystemOneSelector(
        binding=ModelBinding.system_one(model="decision-model-preview", provider=model),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_below_the_thresholds_returns_candidates_untouched() -> None:
    model = _FakeDecisionModel()
    tools = [_tool(i) for i in range(3)]
    skills = [_skill(i) for i in range(2)]

    selected_tools, selected_skills = await _selector(model)(tools, skills, "do something")

    assert selected_tools == tools
    assert selected_skills == skills
    assert model.calls == []


@pytest.mark.asyncio
async def test_tools_above_the_threshold_are_ranked_and_capped() -> None:
    tools = [_tool(i) for i in range(25)]
    model = _FakeDecisionModel(scores={f"tool_{i:02d}": 0.9 for i in range(20, 25)})
    selector = _selector(model, top_k_tools=2, close_read=False)

    selected_tools, selected_skills = await selector(tools, [], "pick a tool")

    # Original order is preserved in the result; ranking only chooses the subset.
    assert [t.name for t in selected_tools] == ["tool_20", "tool_21"]
    assert selected_skills == []
    assert len(model.calls) == 2  # 25 tools over chunks of 20


@pytest.mark.asyncio
async def test_skills_are_ranked_independently() -> None:
    skills = [_skill(i) for i in range(25)]
    model = _FakeDecisionModel(scores={f"skill_{i:02d}": 0.9 for i in range(22, 25)})
    selector = _selector(model, top_k_skills=3, close_read=False)

    _, selected_skills = await selector([], skills, "pick a skill")

    assert [s.name for s in selected_skills] == ["skill_22", "skill_23", "skill_24"]
    assert all(name.startswith("skills_") for call in model.calls for name in call["questions"])


@pytest.mark.asyncio
async def test_only_the_category_above_its_threshold_is_selected() -> None:
    tools = [_tool(i) for i in range(25)]
    skills = [_skill(i) for i in range(2)]
    model = _FakeDecisionModel(scores={f"tool_{i:02d}": 0.9 for i in range(25)})
    selector = _selector(model, top_k_tools=1, close_read=False)

    selected_tools, selected_skills = await selector(tools, skills, "pick")

    assert len(selected_tools) == 1
    assert selected_skills == skills  # 2 is not more than min_skills=20
    assert all(name.startswith("tools_") for call in model.calls for name in call["questions"])


@pytest.mark.asyncio
async def test_close_read_drops_candidates_below_the_fit_threshold_and_uses_full_descriptions() -> None:
    tools = [_tool(i, long_description=f"LONG DETAIL for tool {i}" if i < 3 else None) for i in range(21)]
    # Wide pass: tool_20 wins its chunk outright; tools 0-2 lead the other chunk.
    model = _FakeDecisionModel(
        scores={"tool_20": 0.9, "tool_00": 0.9, "tool_01": 0.9, "tool_02": 0.9},
        fits={"tool_20": 0.05, "tool_00": 0.9, "tool_01": 0.8, "tool_02": 0.7},
    )
    selector = _selector(model, top_k_tools=2, top_k_skills=1, shortlist=4, fit_threshold=0.5)

    selected_tools, _ = await selector(tools, [], "pick a tool")

    # tool_20 reached the shortlist by the wide pass but failed the close read.
    assert [t.name for t in selected_tools] == ["tool_00", "tool_01"]
    assert len(model.calls) == 3  # two wide chunks + one close-read request

    close_questions = model.calls[-1]["questions"]
    choice = next(q for q in close_questions.values() if isinstance(q, Choice))
    assert "LONG DETAIL for tool 0" in choice.criteria["tool_00"]
    assert sum(isinstance(q, Noul) for q in close_questions.values()) == 4


@pytest.mark.asyncio
async def test_close_read_ranks_by_the_choice_and_uses_fit_only_as_a_gate() -> None:
    tools = [_tool(i) for i in range(21)]
    # Both reach the shortlist and pass the fit gate; the Choice prefers tool_01
    # while the Noul prefers tool_00.
    model = _FakeDecisionModel(
        scores={"tool_00": 0.6, "tool_01": 0.9},
        fits={"tool_00": 0.9, "tool_01": 0.6},
    )
    selector = _selector(model, top_k_tools=1, top_k_skills=1, shortlist=2, fit_threshold=0.5)

    selected_tools, _ = await selector(tools, [], "pick a tool")

    assert [t.name for t in selected_tools] == ["tool_01"]


@pytest.mark.asyncio
async def test_all_fits_below_the_threshold_fall_back_to_the_ranking() -> None:
    tools = [_tool(i) for i in range(21)]
    model = _FakeDecisionModel(
        scores={f"tool_{i:02d}": 0.9 for i in range(4)},
        fits={f"tool_{i:02d}": 0.1 for i in range(21)},
    )
    selector = _selector(model, top_k_tools=2, top_k_skills=1, shortlist=4, fit_threshold=0.5)

    selected_tools, _ = await selector(tools, [], "pick")

    # An empty selection would disarm the agent; the ranking is the floor.
    assert len(selected_tools) == 2


@pytest.mark.asyncio
async def test_wide_pass_chunk_size_bounds_each_choice() -> None:
    tools = [_tool(i) for i in range(25)]
    model = _FakeDecisionModel(scores={f"tool_{i:02d}": 1.0 for i in range(25)})
    selector = _selector(model, top_k_tools=2, close_read=False, chunk_size=10)

    await selector(tools, [], "pick")

    assert len(model.calls) == 3
    for call in model.calls:
        (choice,) = call["questions"].values()
        assert isinstance(choice, Choice)
        assert len(choice.criteria) <= 10


@pytest.mark.asyncio
async def test_provider_failure_fails_open_to_the_original_candidates() -> None:
    tools = [_tool(i) for i in range(25)]
    skills = [_skill(i) for i in range(25)]
    model = _FakeDecisionModel()
    model.error = ProviderError("decision model unavailable", retryable=True)
    selector = _selector(model)

    selected_tools, selected_skills = await selector(tools, skills, "pick")

    assert selected_tools == tools
    assert selected_skills == skills


@pytest.mark.asyncio
async def test_an_empty_query_skips_selection() -> None:
    tools = [_tool(i) for i in range(25)]
    model = _FakeDecisionModel()

    selected_tools, _ = await _selector(model)(tools, [], "   ")

    assert selected_tools == tools
    assert model.calls == []


@pytest.mark.asyncio
async def test_selector_timeout_and_binding_defaults_reach_the_provider() -> None:
    tools = [_tool(i) for i in range(25)]
    model = _FakeDecisionModel(scores={f"tool_{i:02d}": 1.0 for i in range(25)})
    selector = _selector(model, top_k_tools=2, close_read=False, timeout=2.5)

    await selector(tools, [], "pick")

    assert model.calls[0]["options"]["timeout"] == 2.5
    assert model.calls[0]["model"] == "decision-model-preview"


@pytest.mark.asyncio
async def test_select_tools_works_as_a_tool_selector_without_thresholds() -> None:
    tools = [_tool(i) for i in range(5)]
    model = _FakeDecisionModel(scores={"tool_00": 0.9, "tool_04": 0.9})

    selected = await _selector(model, min_tools=20, top_k_tools=2, close_read=False).select_tools(tools, "pick")

    assert [t.name for t in selected] == ["tool_00", "tool_04"]


def test_selector_validates_its_configuration() -> None:
    binding = ModelBinding.system_one(model="m", provider=_FakeDecisionModel())

    for kwargs in (
        {"top_k_tools": 0},
        {"top_k_skills": 0},
        {"chunk_size": 0},
        {"summary_chars": 0},
        {"detail_chars": 0},
        {"fit_threshold": 1.5},
        {"fit_threshold": -0.1},
        {"shortlist": 2, "top_k_tools": 3},
        {"shortlist": 16},
    ):
        with pytest.raises(ValueError):
            SystemOneSelector(binding=binding, **kwargs)


def test_selector_rejects_a_binding_for_another_purpose() -> None:
    with pytest.raises(ValueError, match="system_one"):
        SystemOneSelector(binding=ModelBinding.llm(model="mock"))
