"""Live: System One narrows the advertised tool and skill candidates.

Needs a configured decision model: ``CLIVER_SYSTEM_ONE_MODEL`` plus
``CLIVER_SYSTEM_ONE_PROVIDER`` (``bailian`` or ``typesafe``) or
``CLIVER_SYSTEM_ONE_BASE_URL``.  Without it the test warns and skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cliver_core import AgentCore, ModelBinding, ModelBindings, SkillManager, presets
from cliver_core.system_one.selector import SystemOneSelector
from cliver_core.tests.live.support import (
    LIVE_OFF_REASON,
    live_enabled,
    load_env_file,
    skip_unless_keys,
    skip_with_warning,
    tool_names,
)
from cliver_core.tool import tool

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)

TOOL_NAMES = {
    "query_weather",
    "query_stock",
    "convert_currency",
    "book_meeting_room",
    "resize_image",
    "send_email",
    "query_logistics",
    "detect_language",
}
SKILL_NAMES = {
    "weather-brief",
    "finance-report",
    "meeting-notes",
    "image-tools",
    "email-draft",
    "logistics-trace",
}


@tool(name="query_weather", description="Look up the current weather and forecast for a city.")
def query_weather(city: str) -> str:
    return f"{city}: sunny, 22 C."


@tool(name="query_stock", description="Look up a stock quote and daily change by ticker.")
def query_stock(ticker: str) -> str:
    return f"{ticker}: 100.00 (+1.2%)."


@tool(name="convert_currency", description="Convert an amount between two currencies.")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    return f"{amount} {from_currency} = {amount} {to_currency}."


@tool(name="book_meeting_room", description="Reserve a meeting room for a time range.")
def book_meeting_room(room: str, start: str, end: str) -> str:
    return f"{room} booked {start}-{end}."


@tool(name="resize_image", description="Resize an image file to the given width and height.")
def resize_image(path: str, width: int, height: int) -> str:
    return f"{path} resized to {width}x{height}."


@tool(name="send_email", description="Send an email to one recipient with a subject and body.")
def send_email(to: str, subject: str, body: str) -> str:
    return f"Mail queued to {to}: {subject}."


@tool(name="query_logistics", description="Track a parcel by tracking number and show its checkpoints.")
def query_logistics(tracking_number: str) -> str:
    return f"{tracking_number}: in transit."


@tool(name="detect_language", description="Detect the language of a piece of text.")
def detect_language(text: str) -> str:
    return "en"


TOOLS = [
    query_weather,
    query_stock,
    convert_currency,
    book_meeting_room,
    resize_image,
    send_email,
    query_logistics,
    detect_language,
]

_SKILL_BODIES = {
    "weather-brief": "Use for weather summaries and trip forecasts.",
    "finance-report": "Use for stock quotes, portfolio summaries, and currency conversion.",
    "meeting-notes": "Use to turn a transcript into meeting notes and action items.",
    "image-tools": "Use to resize, crop, or convert image files.",
    "email-draft": "Use to draft and send concise emails.",
    "logistics-trace": "Use to trace parcels and summarize delivery checkpoints.",
}


def _write_skills(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    for name, description in _SKILL_BODIES.items():
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n{description}\n",
            encoding="utf-8",
        )
    return root


def _system_one_binding() -> ModelBinding:
    load_env_file()
    values = os.environ
    model = values.get("CLIVER_SYSTEM_ONE_MODEL", "").strip()
    provider = values.get("CLIVER_SYSTEM_ONE_PROVIDER", "").strip().lower()
    base_url = values.get("CLIVER_SYSTEM_ONE_BASE_URL", "").strip()
    if not model and not provider and not base_url:
        skip_with_warning(
            "live verification skipped: System One is not configured; set "
            "CLIVER_SYSTEM_ONE_PROVIDER=bailian (with CLIVER_SYSTEM_ONE_WORKSPACE_ID) or "
            "typesafe, or CLIVER_SYSTEM_ONE_BASE_URL plus CLIVER_SYSTEM_ONE_MODEL to run this check"
        )

    if provider == "laya":
        skip_with_warning(
            "live verification skipped: the laya System One backend needs the local "
            "extra (make init-local) and a model download"
        )
    if provider == "bailian":
        workspace_id = values.get("CLIVER_SYSTEM_ONE_WORKSPACE_ID", "").strip()
        if not workspace_id and not base_url:
            skip_with_warning(
                "live verification skipped: set CLIVER_SYSTEM_ONE_WORKSPACE_ID "
                "(or CLIVER_SYSTEM_ONE_BASE_URL) for the bailian provider"
            )
        explicit_key = values.get("CLIVER_SYSTEM_ONE_API_KEY", "").strip() or None
        if explicit_key is None:
            skip_unless_keys("DASHSCOPE_API_KEY")
        bailian_options = {"model": model} if model else {}
        return presets.bailian_system_one(
            workspace_id=workspace_id,
            api_key=explicit_key,
            region=values.get("CLIVER_SYSTEM_ONE_REGION", "").strip() or "cn-beijing",
            base_url=base_url,
            **bailian_options,
        )
    if provider == "typesafe":
        explicit_key = values.get("CLIVER_SYSTEM_ONE_API_KEY", "").strip() or None
        if explicit_key is None:
            skip_unless_keys("TYPESAFE_API_KEY")
        typesafe_options = {"model": model} if model else {}
        return presets.typesafe_system_one(api_key=explicit_key, base_url=base_url, **typesafe_options)
    if provider:
        skip_with_warning(f"live verification skipped: unknown CLIVER_SYSTEM_ONE_PROVIDER {provider!r}")
    if not base_url:
        skip_with_warning("live verification skipped: a custom System One endpoint needs CLIVER_SYSTEM_ONE_BASE_URL")
    if not model:
        skip_with_warning("live verification skipped: a custom System One endpoint needs CLIVER_SYSTEM_ONE_MODEL")
    skip_unless_keys("CLIVER_SYSTEM_ONE_API_KEY")
    return ModelBinding.system_one(
        model=model,
        api_key=os.environ["CLIVER_SYSTEM_ONE_API_KEY"],
        base_url=base_url,
    )


async def test_system_one_filters_tools_and_skills(tmp_path: Path) -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    selector = SystemOneSelector(
        binding=_system_one_binding(),
        min_tools=1,
        min_skills=1,
        top_k_tools=3,
        top_k_skills=2,
        chunk_size=10,
        shortlist=3,
        close_read=True,
        timeout=30,
    )
    agent = AgentCore(
        ModelBindings(llm=presets.deepseek(temperature=0)),
        builtin_tools=TOOLS,
        skill_manager=SkillManager([_write_skills(tmp_path)]),
        candidate_selector=selector,
        run_timeout=120,
    )

    try:
        response = await agent.chat(
            "What is the weather in Beijing right now? Use a tool if one fits.",
            run_id="live-system-one",
        )
        assert response.message.content
        assert response.run_id == "live-system-one"
        trace = response.message.trace or []
        advertised = tool_names(trace[0])
        regular = advertised - {"FindSkills", "ActivateSkill"}
        assert len(regular) == selector.top_k_tools
        assert regular <= TOOL_NAMES

        system_prompt = trace[0]["request"]["system_prompt"] or ""
        visible_skills = {name for name in SKILL_NAMES if name in system_prompt}
        assert len(visible_skills) == selector.top_k_skills
    finally:
        await agent.aclose()
        await selector.binding.aclose()
