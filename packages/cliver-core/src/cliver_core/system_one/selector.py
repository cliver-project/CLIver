"""System One ranking for large tool and skill catalogues.

Sending every tool and skill to the main model is expensive and dilutes its
attention.  The selector asks a System One decision model which candidates
fit the user's request, then advertises only the best ones.

Two stages, after the TypeSafe ``skill_suggestion`` cookbook and fastmcp's
``jev_search``:

1. **Wide pass** — candidates are chunked (``chunk_size``, default 20; local
   ``laya`` degrades past roughly 20 options) and each chunk gets one
   ``Choice`` over short summaries.  Merge by probability.
2. **Close read** — the top ``shortlist`` candidates get one ``Choice`` over
   full details plus one ``Noul`` per candidate ("would using this help?").
   The Choice ranks the survivors; the Noul is a relevance gate, and
   candidates below ``fit_threshold`` are dropped.  If that would drop
   everything, the ranking stands — a selection failure must never disarm
   the agent.

The selector is fail-open: any provider error, timeout, or malformed answer
returns the original candidates unchanged.  It never adds candidates and
never reorders the ones it keeps (the result preserves the caller's order,
so behaviour stays deterministic across runs).

Use it as the agent's cross-category guardrail::

    from cliver_core.system_one.selector import SystemOneSelector

    selector = SystemOneSelector(binding=presets.bailian_system_one(workspace_id="..."))
    agent = AgentCore(bindings, candidate_selector=selector)

``select_tools`` and ``select_skills`` are also usable directly as
``tool_selector`` / ``skill_selector`` (they skip the activation thresholds).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from cliver_core.bindings import ModelBinding
from cliver_core.errors import ProviderError
from cliver_core.skill import Skill
from cliver_core.system_one.types import Choice, ChoiceAnswer, Noul, NoulAnswer, Question, SystemOneResult
from cliver_core.tool import CLIverTool

__all__ = ["SystemOneSelector"]

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    index: int
    name: str
    summary: str
    detail: str
    wide: float = 0.0
    fit: float | None = None
    choice_score: float = 0.0


def _one_line(text: str, limit: int) -> str:
    """Collapse whitespace and truncate — criteria values are one-line text."""
    return " ".join((text or "").split())[:limit]


class SystemOneSelector:
    """Rank tools and skills with a System One decision model.

    Args:
        binding: A ``ModelBinding.system_one(...)`` (or a local system-one
            binding).  The selector calls its provider directly, so the
            binding's default call options (``path``, ``timeout``, …) apply.
        min_tools: Activate tool selection only when more than this many
            tools are candidates.
        min_skills: The same threshold for skills.
        top_k_tools: Maximum tools to keep.
        top_k_skills: Maximum skills to keep.
        chunk_size: Wide-pass candidates per ``Choice`` question.
        shortlist: Candidates entering the close read, per category.  Must
            cover ``top_k_tools`` / ``top_k_skills`` and stay at most 15 —
            the close read sends one ``Choice`` plus one ``Noul`` per
            candidate, and the decision-model APIs cap a request at 16
            questions.
        close_read: Run the second stage.  Disable to trade accuracy for
            one request per chunk.
        fit_threshold: Minimum ``Noul`` probability for a close-read
            candidate to survive.
        summary_chars: Wide-pass description length per candidate.
        detail_chars: Close-read description length per candidate.
        timeout: Request timeout (seconds) overriding the binding default.
    """

    def __init__(
        self,
        *,
        binding: ModelBinding,
        min_tools: int = 20,
        min_skills: int = 20,
        top_k_tools: int = 10,
        top_k_skills: int = 5,
        chunk_size: int = 20,
        shortlist: int = 12,
        close_read: bool = True,
        fit_threshold: float = 0.3,
        summary_chars: int = 120,
        detail_chars: int = 600,
        timeout: float | None = None,
    ) -> None:
        if not isinstance(binding, ModelBinding):
            raise TypeError("SystemOneSelector requires a ModelBinding")
        if binding.purpose not in (None, "system_one"):
            raise ValueError(
                f"SystemOneSelector needs a system_one binding, got purpose '{binding.purpose}' — "
                "build it with ModelBinding.system_one(...) or a system-one preset."
            )
        for name, value, minimum in (
            ("min_tools", min_tools, 0),
            ("min_skills", min_skills, 0),
            ("top_k_tools", top_k_tools, 1),
            ("top_k_skills", top_k_skills, 1),
            ("chunk_size", chunk_size, 1),
            ("shortlist", shortlist, 1),
            ("summary_chars", summary_chars, 1),
            ("detail_chars", detail_chars, 1),
        ):
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum} (got {value})")
        if not 0.0 <= fit_threshold <= 1.0:
            raise ValueError(f"fit_threshold must be in [0, 1] (got {fit_threshold})")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive (got {timeout})")
        if close_read:
            if shortlist < max(top_k_tools, top_k_skills):
                raise ValueError(
                    f"shortlist must cover top_k_tools/top_k_skills "
                    f"(got shortlist={shortlist}, top_k={max(top_k_tools, top_k_skills)})"
                )
            if shortlist > 15:
                raise ValueError(
                    "shortlist must be at most 15 — the close read sends one Choice plus one "
                    "Noul per candidate, and decision-model APIs cap a request at 16 questions"
                )

        self.binding = binding
        self.min_tools = min_tools
        self.min_skills = min_skills
        self.top_k_tools = top_k_tools
        self.top_k_skills = top_k_skills
        self.chunk_size = chunk_size
        self.shortlist = shortlist
        self.close_read = close_read
        self.fit_threshold = fit_threshold
        self.summary_chars = summary_chars
        self.detail_chars = detail_chars
        self.timeout = timeout

    # ── AgentCore selector surfaces ───────────────────────────

    async def __call__(
        self,
        tools: Sequence[CLIverTool],
        skills: Sequence[Skill],
        query: str,
    ) -> tuple[list[CLIverTool], list[Skill]]:
        """``candidate_selector`` — activate each category above its threshold."""
        tools = list(tools)
        skills = list(skills)
        selected_tools = await self.select_tools(tools, query) if len(tools) > self.min_tools else tools
        selected_skills = await self.select_skills(skills, query) if len(skills) > self.min_skills else skills
        return selected_tools, selected_skills

    async def select_tools(self, tools: Sequence[CLIverTool], query: str) -> list[CLIverTool]:
        """Rank *tools* and keep up to ``top_k_tools`` — fail-open."""
        candidates = list(tools)
        try:
            return await self._select("tools", candidates, query, self.top_k_tools)
        except Exception as exc:
            logger.warning("System One tool selection failed; keeping all %d tools: %s", len(candidates), exc)
            return candidates

    async def select_skills(self, skills: Sequence[Skill], query: str) -> list[Skill]:
        """Rank *skills* and keep up to ``top_k_skills`` — fail-open."""
        candidates = list(skills)
        try:
            return await self._select("skills", candidates, query, self.top_k_skills)
        except Exception as exc:
            logger.warning("System One skill selection failed; keeping all %d skills: %s", len(candidates), exc)
            return candidates

    # ── Selection ────────────────────────────────────────────

    async def _select(
        self,
        kind: str,
        candidates: Sequence[Any],
        query: str,
        top_k: int,
    ) -> list[Any]:
        if not (query or "").strip() or len(candidates) <= top_k:
            return list(candidates)

        entries = self._entries(kind, candidates)
        if len({entry.name for entry in entries}) != len(entries):
            logger.warning("System One %s selection skipped: duplicate candidate names", kind)
            return list(candidates)

        await self._wide_pass(kind, entries, query)
        ordered = sorted(entries, key=lambda entry: (-entry.wide, entry.index))[: self.shortlist]

        if self.close_read:
            ordered = await self._close_read(kind, ordered, query)
            kept = [entry for entry in ordered if entry.fit is not None and entry.fit >= self.fit_threshold]
            ordered = kept or ordered

        chosen = sorted(entry.index for entry in ordered[:top_k])
        return [candidates[index] for index in chosen]

    def _entries(self, kind: str, candidates: Sequence[Any]) -> list[_Entry]:
        entries: list[_Entry] = []
        for index, candidate in enumerate(candidates):
            if kind == "tools":
                description = candidate.description or ""
                detail_source = candidate.long_description or description
            else:
                description = candidate.description or ""
                detail_source = f"{description}\n\n{candidate.body}" if candidate.body else description
            entries.append(
                _Entry(
                    index=index,
                    name=candidate.name,
                    summary=_one_line(description, self.summary_chars),
                    detail=_one_line(detail_source, self.detail_chars),
                )
            )
        return entries

    async def _wide_pass(self, kind: str, entries: Sequence[_Entry], query: str) -> None:
        """One ``Choice`` per chunk; merge probabilities into ``entry.wide``."""
        singular = kind[:-1]
        for chunk_index, start in enumerate(range(0, len(entries), self.chunk_size)):
            chunk = list(entries[start : start + self.chunk_size])
            question_name = f"{kind}_{chunk_index}"
            question = Choice(
                instructions=(
                    f"User request:\n{query}\n\n"
                    f"Which {singular} is most likely needed to fulfill this request? "
                    "Choose the closest option even if none is a perfect match."
                ),
                criteria={entry.name: entry.summary for entry in chunk},
            )
            result = await self._predict(query, {question_name: question})
            answer = result.answers.get(question_name)
            if not isinstance(answer, ChoiceAnswer):
                raise ProviderError(f"System One returned no choice answer for '{question_name}'")
            for entry in chunk:
                entry.wide = float(answer.probabilities.get(entry.name, 0.0))

    async def _close_read(
        self,
        kind: str,
        shortlist: Sequence[_Entry],
        query: str,
    ) -> list[_Entry]:
        """One ``Choice`` over full details plus one ``Noul`` per candidate."""
        singular = kind[:-1]
        choice_name = f"{kind}_close"
        questions: dict[str, Question] = {
            choice_name: Choice(
                instructions=(
                    f"User request:\n{query}\n\n"
                    f"Which {singular} is most likely needed to fulfill this request? "
                    "Compare the detailed descriptions and choose the best match."
                ),
                criteria={entry.name: entry.detail for entry in shortlist},
            )
        }
        for position, entry in enumerate(shortlist):
            questions[f"{kind}_fit_{position}"] = Noul(
                instructions=(
                    f"User request:\n{query}\n\n"
                    f"Would using the {singular} '{entry.name}' help fulfill this request?\n\n"
                    f"{entry.detail}"
                )
            )

        result = await self._predict(query, questions)
        choice_answer = result.answers.get(choice_name)
        if not isinstance(choice_answer, ChoiceAnswer):
            raise ProviderError(f"System One returned no choice answer for '{choice_name}'")
        for position, entry in enumerate(shortlist):
            fit_answer = result.answers.get(f"{kind}_fit_{position}")
            if not isinstance(fit_answer, NoulAnswer):
                raise ProviderError(f"System One returned no noul answer for '{kind}_fit_{position}'")
            entry.fit = float(fit_answer.noul)
            entry.choice_score = float(choice_answer.probabilities.get(entry.name, 0.0))

        # The Choice over full details is the ranking; the per-candidate Noul is
        # only the relevance gate applied by the caller.  Ranking by ``fit``
        # instead would let an uncalibrated yes/no answer reorder strong
        # candidates out of the top-k (observed with local laya).
        return sorted(shortlist, key=lambda entry: (-entry.choice_score, entry.index))

    async def _predict(self, query: str, questions: dict[str, Question]) -> SystemOneResult:
        options = dict(self.binding.call_defaults())
        if self.timeout is not None:
            options["timeout"] = self.timeout
        async with self.binding.lease_provider() as provider:
            return await provider.system_one(
                state=query,
                questions=questions,
                model=self.binding.model,
                **options,
            )
