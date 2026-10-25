"""Deterministic sequential pipelines of skill runs (sub-agent style).

The caller defines the ordered steps; no LLM decides the sequence.  Each
step runs in its own fresh context — a skill via ``AgentCore.run_skill()``,
or a raw prompt via ``AgentCore.chat()`` — and the next step's input is
composed from the original input and the prior results.  A synthesis step
is an ordinary final step whose input builder reads the earlier outputs.

AgentCore stays stateless: the pipeline holds only step definitions, the
default agent, and the event handler, so one pipeline can be run
concurrently.  No fan-out, branching, or DAG is provided in v1; the step
model is an ordered list, which leaves room for those later.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from cliver_core.agent import AgentCore
from cliver_core.events import EventHandler, InferenceEvent, PipelineEvent, PipelineEventType, ToolEvent
from cliver_core.messages import UsageInfo
from cliver_core.provider import CLIverResponse
from cliver_core.skill import Skill

__all__ = [
    "FailurePolicy",
    "InputBuilder",
    "PipelineContext",
    "PipelineResult",
    "PipelineStep",
    "SkillPipeline",
    "StepOutput",
    "StepResult",
    "StepRunResult",
    "StepRunner",
]


class FailurePolicy(str, Enum):
    """What the pipeline does when a step raises."""

    STOP = "stop"  # fail fast: the remaining steps do not run (default)
    CONTINUE = "continue"  # record the error and run the next step


@dataclass(frozen=True)
class StepOutput:
    """A runner's output for one step, before the pipeline names it."""

    content: str
    metadata: dict[str, Any] | None = None
    usage: UsageInfo | None = None


StepRunResult = StepOutput | CLIverResponse | str


@dataclass(frozen=True)
class StepResult:
    """One executed step's outcome.  A failed step has ``error`` set and no content."""

    name: str
    content: str = ""
    metadata: dict[str, Any] | None = None
    usage: UsageInfo | None = None
    error: str | None = None


@dataclass(frozen=True)
class PipelineContext:
    """The original input plus the ordered results of the steps before this one."""

    input: str
    results: tuple[StepResult, ...] = ()

    def result(self, name: str) -> StepResult | None:
        """Return the prior result named *name*, or ``None``."""
        return next((result for result in self.results if result.name == name), None)

    @property
    def previous(self) -> StepResult | None:
        """The immediately preceding result, or ``None`` for the first step."""
        return self.results[-1] if self.results else None


InputBuilder = Callable[[PipelineContext], "str | Awaitable[str]"]

#: ``(step, step_input, context, emit) -> StepOutput | CLIverResponse | str``.
#: ``emit`` accepts :class:`~cliver_core.events.ToolEvent` /
#: :class:`~cliver_core.events.PipelineEvent` values and forwards them to the
#: pipeline's handler.  A runner using an ``AgentCore`` should pass
#: ``on_event=emit`` so the step's tool events are forwarded.
StepRunner = Callable[["PipelineStep", str, PipelineContext, EventHandler], Awaitable[StepRunResult]]


@dataclass(frozen=True)
class PipelineStep:
    """One step of a pipeline.

    ``skill`` runs a skill (name or loaded :class:`~cliver_core.skill.Skill`);
    ``prompt`` is raw step instructions, appended to the single system
    message — as extra instructions alongside a skill, or as the whole
    instruction when no skill is given.  ``input_builder`` composes this
    step's user input from the :class:`PipelineContext` (default: the
    pipeline input).  ``runner`` overrides the pipeline's default runner for
    this step — different steps may use different ``AgentCore`` instances,
    models, tools, or skill sets.  ``on_failure`` defaults to stopping the
    pipeline.

    ``input_builder`` may be sync or async; ``runner`` is async.
    """

    name: str
    skill: Skill | str | None = None
    prompt: str | None = None
    input_builder: InputBuilder | None = None
    runner: StepRunner | None = None
    on_failure: FailurePolicy = FailurePolicy.STOP

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("pipeline step name must not be empty")
        if self.skill is None and self.prompt is None:
            raise ValueError(f"pipeline step '{self.name}' needs a skill or prompt")
        if not isinstance(self.on_failure, FailurePolicy):
            object.__setattr__(self, "on_failure", FailurePolicy(self.on_failure))


@dataclass(frozen=True)
class PipelineResult:
    """The ordered results of one run, plus whether a failure stopped it."""

    results: tuple[StepResult, ...] = ()
    stopped: bool = False

    @property
    def ok(self) -> bool:
        """True when every step ran and none failed."""
        return not self.stopped and all(result.error is None for result in self.results)

    @property
    def final(self) -> StepResult | None:
        """The last executed result (the synthesis step's output in the usual shape)."""
        return self.results[-1] if self.results else None


class SkillPipeline:
    """Run ordered steps sequentially, each in a fresh context.

    ``agent`` supplies the default runner (``run_skill`` for skill steps,
    ``chat`` for prompt-only steps); pass ``agent=None`` only when every
    step injects its own ``runner``.  ``on_event`` receives pipeline
    lifecycle events and every forwarded tool event.
    """

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        *,
        agent: AgentCore | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        self.steps = tuple(steps)
        self.agent = agent
        self.on_event = on_event

        names = [step.name for step in self.steps]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate step name(s): {', '.join(duplicates)}")
        without_runner = [step.name for step in self.steps if step.runner is None]
        if without_runner and agent is None:
            raise ValueError("SkillPipeline needs agent=... for steps without a runner: " + ", ".join(without_runner))

    async def run(self, input: str) -> PipelineResult:
        """Run the steps in order.  Cancellation emits ``pipeline.end`` and re-raises."""
        started = time.monotonic()
        await self._emit(PipelineEvent(event=PipelineEventType.START, data={"step_count": len(self.steps)}))

        results: list[StepResult] = []
        stopped = False
        try:
            for index, step in enumerate(self.steps):
                result = await self._run_step(step, index, input, tuple(results))
                results.append(result)
                if result.error is not None and step.on_failure is FailurePolicy.STOP:
                    stopped = True
                    break
        except asyncio.CancelledError:
            await self._emit(
                PipelineEvent(
                    event=PipelineEventType.END,
                    data={"cancelled": True, "steps_run": len(results)},
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            )
            raise

        ok = not stopped and all(result.error is None for result in results)
        await self._emit(
            PipelineEvent(
                event=PipelineEventType.END,
                data={"ok": ok, "stopped": stopped, "steps_run": len(results)},
                duration_ms=(time.monotonic() - started) * 1000,
            )
        )
        return PipelineResult(results=tuple(results), stopped=stopped)

    async def _run_step(
        self,
        step: PipelineStep,
        index: int,
        pipeline_input: str,
        prior: tuple[StepResult, ...],
    ) -> StepResult:
        context = PipelineContext(input=pipeline_input, results=prior)
        started = time.monotonic()
        await self._emit(PipelineEvent(event=PipelineEventType.STEP_START, step_name=step.name, step_index=index))
        try:
            step_input = pipeline_input
            if step.input_builder is not None:
                built = step.input_builder(context)
                step_input = await built if inspect.isawaitable(built) else built
            runner = step.runner or self._default_runner
            output = await runner(step, step_input, context, self._emit)
            result = _as_step_result(step.name, output)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit(
                PipelineEvent(
                    event=PipelineEventType.STEP_ERROR,
                    step_name=step.name,
                    step_index=index,
                    error=str(exc),
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            )
            return StepResult(name=step.name, error=str(exc))

        await self._emit(
            PipelineEvent(
                event=PipelineEventType.STEP_END,
                step_name=step.name,
                step_index=index,
                data={
                    "content_chars": len(result.content),
                    "usage": result.usage.model_dump() if result.usage else None,
                },
                duration_ms=(time.monotonic() - started) * 1000,
            )
        )
        return result

    async def _default_runner(
        self,
        step: PipelineStep,
        step_input: str,
        context: PipelineContext,
        emit: EventHandler,
    ) -> CLIverResponse:
        assert self.agent is not None  # guaranteed by __init__
        if step.skill is not None:
            return await self.agent.run_skill(
                step.skill,
                step_input,
                system_prompt=step.prompt,
                on_event=emit,
            )
        return await self.agent.chat(step_input, system_prompt=step.prompt, on_event=emit)

    async def _emit(self, event: InferenceEvent | ToolEvent | PipelineEvent) -> None:
        if self.on_event:
            await self.on_event(event)


def _as_step_result(name: str, output: StepRunResult) -> StepResult:
    if isinstance(output, StepOutput):
        return StepResult(name=name, content=output.content, metadata=output.metadata, usage=output.usage)
    if isinstance(output, CLIverResponse):
        content = output.message.content
        return StepResult(name=name, content=content if isinstance(content, str) else "", usage=output.usage)
    return StepResult(name=name, content=output)
