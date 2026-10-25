"""Deterministic sequential skill pipelines — fake runners, no network."""

from __future__ import annotations

import asyncio

import pytest

from cliver_core.events import PipelineEvent, PipelineEventType, ToolEvent, ToolEventType
from cliver_core.messages import CLIverMessage, UsageInfo
from cliver_core.pipeline import (
    FailurePolicy,
    PipelineContext,
    PipelineStep,
    SkillPipeline,
    StepOutput,
    StepResult,
)
from cliver_core.provider import CLIverResponse


async def _noop_runner(step, step_input, context, emit):
    return StepOutput(content=step_input)


async def test_steps_run_in_order_and_compose_inputs_from_prior_results():
    seen: list[tuple[str, str, list[str]]] = []

    async def runner(step, step_input, context, emit):
        seen.append((step.name, step_input, [result.name for result in context.results]))
        return StepOutput(content=f"{step.name}:{step_input}")

    async def previous_content(context: PipelineContext) -> str:
        return context.previous.content

    pipeline = SkillPipeline(
        [
            PipelineStep("market", prompt="market", runner=runner),
            PipelineStep(
                "competitors",
                prompt="competitors",
                runner=runner,
                input_builder=lambda context: f"{context.input}+{context.result('market').content}",
            ),
            PipelineStep("economics", prompt="economics", runner=runner, input_builder=previous_content),
        ]
    )

    result = await pipeline.run("idea")

    assert seen == [
        ("market", "idea", []),
        ("competitors", "idea+market:idea", ["market"]),
        ("economics", "competitors:idea+market:idea", ["market", "competitors"]),
    ]
    assert [step.content for step in result.results] == [
        "market:idea",
        "competitors:idea+market:idea",
        "economics:competitors:idea+market:idea",
    ]
    assert result.ok is True
    assert result.stopped is False
    assert result.final.name == "economics"


async def test_step_output_metadata_and_usage_are_collected():
    usage = UsageInfo(input_tokens=11, output_tokens=7)

    async def runner(step, step_input, context, emit):
        return StepOutput(content="report", metadata={"score": 0.9}, usage=usage)

    result = await SkillPipeline([PipelineStep("one", prompt="p", runner=runner)]).run("x")

    assert result.final == StepResult(name="one", content="report", metadata={"score": 0.9}, usage=usage)


async def test_runner_may_return_a_plain_string():
    async def runner(step, step_input, context, emit):
        return "plain answer"

    result = await SkillPipeline([PipelineStep("one", prompt="p", runner=runner)]).run("x")

    assert result.final == StepResult(name="one", content="plain answer")


class _FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.usage = UsageInfo(input_tokens=3, output_tokens=2)

    async def run_skill(self, skill, user_input, **kwargs):
        self.calls.append(("run_skill", skill, user_input, kwargs))
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content=f"ran {skill}"),
            usage=self.usage,
        )

    async def chat(self, user_input, **kwargs):
        self.calls.append(("chat", user_input, kwargs))
        return CLIverResponse(message=CLIverMessage(role="assistant", content="raw answer"), usage=self.usage)


async def test_default_runner_uses_run_skill_chat_and_per_step_overrides():
    agent = _FakeAgent()

    async def custom(step, step_input, context, emit):
        return StepOutput(content="custom answer")

    pipeline = SkillPipeline(
        [
            PipelineStep("skilled", skill="market", prompt="Extra instruction."),
            PipelineStep("raw", prompt="Answer directly."),
            PipelineStep("custom", prompt="p", runner=custom),
        ],
        agent=agent,
    )

    result = await pipeline.run("idea")

    assert [call[0] for call in agent.calls] == ["run_skill", "chat"]
    skill_call = agent.calls[0]
    assert skill_call[1] == "market"
    assert skill_call[2] == "idea"
    assert skill_call[3]["system_prompt"] == "Extra instruction."
    assert callable(skill_call[3]["on_event"])
    chat_call = agent.calls[1]
    assert chat_call[1] == "idea"
    assert chat_call[2]["system_prompt"] == "Answer directly."
    assert callable(chat_call[2]["on_event"])
    assert [step.content for step in result.results] == ["ran market", "raw answer", "custom answer"]
    assert result.results[0].usage is agent.usage


async def test_failure_stops_after_the_failing_step_by_default():
    ran: list[str] = []

    async def boom(step, step_input, context, emit):
        raise RuntimeError("market failed")

    async def later(step, step_input, context, emit):
        ran.append(step.name)
        return StepOutput(content="unreachable")

    result = await SkillPipeline(
        [
            PipelineStep("market", prompt="p", runner=boom),
            PipelineStep("economics", prompt="p", runner=later),
        ]
    ).run("x")

    assert ran == []
    assert result.stopped is True
    assert result.ok is False
    assert result.results == (StepResult(name="market", error="market failed"),)
    assert result.final.error == "market failed"


async def test_continue_runs_later_steps_and_exposes_the_error_in_context():
    seen: dict = {}

    async def boom(step, step_input, context, emit):
        raise RuntimeError("bad")

    async def later(step, step_input, context, emit):
        seen["error"] = context.result("market").error
        seen["results"] = [result.name for result in context.results]
        return StepOutput(content="final")

    result = await SkillPipeline(
        [
            PipelineStep("market", prompt="p", runner=boom, on_failure=FailurePolicy.CONTINUE),
            PipelineStep("economics", prompt="p", runner=later),
        ]
    ).run("x")

    assert seen == {"error": "bad", "results": ["market"]}
    assert result.stopped is False
    assert result.ok is False
    assert [step.name for step in result.results] == ["market", "economics"]
    assert result.final.content == "final"


async def test_empty_pipeline_returns_a_successful_empty_result():
    events: list = []

    async def on_event(event) -> None:
        events.append(event)

    result = await SkillPipeline([], on_event=on_event).run("x")

    assert result.results == ()
    assert result.ok is True
    assert result.stopped is False
    assert result.final is None
    assert [event.event for event in events] == [PipelineEventType.START, PipelineEventType.END]
    assert events[0].data == {"step_count": 0}


async def test_events_are_ordered_and_tool_events_are_forwarded():
    events: list = []

    async def on_event(event) -> None:
        events.append(event)

    async def runner(step, step_input, context, emit):
        tool_event = ToolEvent(event=ToolEventType.START, tool_name="Read", tool_call_id=f"{step.name}-1")
        await emit(tool_event)
        await emit(ToolEvent(event=ToolEventType.END, tool_name="Read", tool_call_id=f"{step.name}-1"))
        return StepOutput(content=step.name)

    await SkillPipeline(
        [
            PipelineStep("one", prompt="p", runner=runner),
            PipelineStep("two", prompt="p", runner=runner),
        ],
        on_event=on_event,
    ).run("x")

    assert [(type(event), event.event) for event in events] == [
        (PipelineEvent, PipelineEventType.START),
        (PipelineEvent, PipelineEventType.STEP_START),
        (ToolEvent, ToolEventType.START),
        (ToolEvent, ToolEventType.END),
        (PipelineEvent, PipelineEventType.STEP_END),
        (PipelineEvent, PipelineEventType.STEP_START),
        (ToolEvent, ToolEventType.START),
        (ToolEvent, ToolEventType.END),
        (PipelineEvent, PipelineEventType.STEP_END),
        (PipelineEvent, PipelineEventType.END),
    ]
    step_events = [event for event in events if isinstance(event, PipelineEvent)]
    assert [event.step_name for event in step_events] == [None, "one", "one", "two", "two", None]
    assert step_events[-1].data == {"ok": True, "stopped": False, "steps_run": 2}


async def test_cancellation_emits_pipeline_end_and_reraises():
    events: list = []

    async def on_event(event) -> None:
        events.append(event)

    async def runner(step, step_input, context, emit):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await SkillPipeline([PipelineStep("one", prompt="p", runner=runner)], on_event=on_event).run("x")

    pipeline_events = [event for event in events if isinstance(event, PipelineEvent)]
    assert [event.event for event in pipeline_events] == [
        PipelineEventType.START,
        PipelineEventType.STEP_START,
        PipelineEventType.END,
    ]
    assert pipeline_events[-1].data["cancelled"] is True


def test_pipeline_rejects_duplicate_step_names():
    with pytest.raises(ValueError, match="duplicate step name"):
        SkillPipeline(
            [
                PipelineStep("one", prompt="p", runner=_noop_runner),
                PipelineStep("one", prompt="p", runner=_noop_runner),
            ]
        )


def test_pipeline_step_requires_a_skill_or_prompt():
    with pytest.raises(ValueError, match="skill or prompt"):
        PipelineStep("one")


def test_pipeline_requires_an_agent_for_steps_without_a_runner():
    with pytest.raises(ValueError, match="agent"):
        SkillPipeline([PipelineStep("one", prompt="p")])
