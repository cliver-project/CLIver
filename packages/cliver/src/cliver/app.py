"""Application-side wiring for the CLI.

The core stays config-free: this module supplies CLIver's conventions — the
eight skill sources, the builtin toolset, the identity prompt, the model
endpoint from environment variables, and the interactive permission callback.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from cliver_core import AgentCore, Decision, ModelBinding, ModelBindings, ToolCallContext, presets
from cliver_core.events import InferenceEvent, InferenceEventType, ToolEvent, ToolEventType
from cliver_core.local import LocalSystemOneProvider, local_system_one
from cliver_core.provider.providers import API_KEY_ENV
from cliver_core.system_one.selector import SystemOneSelector
from cliver_core.tools import discover_builtin_tools
from dotenv import load_dotenv

from cliver.paths import get_config_dir
from cliver.skills import create_skill_manager

__all__ = [
    "build_agent",
    "llm_binding_from_env",
    "load_environment",
    "make_event_logger",
    "make_permission_callback",
    "selector_from_env",
    "system_one_binding_from_env",
]


def load_environment(config_dir: Path | None = None) -> None:
    """Load ``{config_dir}/.env`` without overriding the process environment."""
    config_path = get_config_dir() if config_dir is None else Path(config_dir).expanduser()
    env_file = config_path / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def llm_binding_from_env(
    env: Mapping[str, str] | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    protocol: str | None = None,
    api_key: str | None = None,
) -> ModelBinding:
    """Build the chat binding from explicit values, then ``CLIVER_*`` variables.

    A known ``provider`` supplies its default ``base_url``; otherwise an
    explicit ``base_url`` is required.  Without an explicit key or
    ``CLIVER_API_KEY``, the key comes from the brand's conventional
    environment variable (``DEEPSEEK_API_KEY``, ``MINIMAX_API_KEY``, ...).
    """
    values = os.environ if env is None else env

    resolved_model = model or values.get("CLIVER_MODEL", "")
    if not resolved_model:
        raise ValueError("No chat model configured: pass --model or set CLIVER_MODEL.")

    resolved_provider = provider if provider is not None else values.get("CLIVER_PROVIDER", "")
    resolved_base_url = base_url if base_url is not None else values.get("CLIVER_BASE_URL", "")
    if not resolved_provider and not resolved_base_url:
        raise ValueError(
            "No endpoint configured: pass --provider or --base-url (or set CLIVER_PROVIDER / CLIVER_BASE_URL)."
        )

    resolved_protocol = protocol or values.get("CLIVER_PROTOCOL", "openai")
    resolved_api_key = api_key if api_key is not None else values.get("CLIVER_API_KEY", "")
    if not resolved_api_key:
        brand = (resolved_provider or "").strip().lower()
        resolved_api_key = next(
            (values[name] for name in API_KEY_ENV.get(brand, ()) if values.get(name)),
            "",
        )

    return ModelBinding.llm(
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        protocol=resolved_protocol,
        provider_name=resolved_provider or None,
    )


def system_one_binding_from_env(env: Mapping[str, str] | None = None) -> ModelBinding | None:
    """Build the System One decision binding from ``CLIVER_SYSTEM_ONE_*``.

    Returns ``None`` when no System One model is configured.  ``PROVIDER`` is
    ``bailian`` (needs ``WORKSPACE_ID``; ``REGION`` defaults to
    ``cn-beijing``), ``typesafe``, or ``laya`` (a local checkpoint, default
    backend ``laya``; ``DEVICE`` and ``MAX_LEN`` tune it).  A custom HTTP
    endpoint using the shared ``/v1/systemone`` wire sets ``BASE_URL`` and
    leaves ``PROVIDER`` unset.

    The key is ``CLIVER_SYSTEM_ONE_API_KEY`` or the provider's conventional
    variable (``DASHSCOPE_API_KEY`` / ``TYPESAFE_API_KEY``).
    """
    values = os.environ if env is None else env
    model = values.get("CLIVER_SYSTEM_ONE_MODEL", "").strip()
    provider = values.get("CLIVER_SYSTEM_ONE_PROVIDER", "").strip().lower()
    base_url = values.get("CLIVER_SYSTEM_ONE_BASE_URL", "").strip()
    if not model and not provider and not base_url:
        return None
    if not model:
        raise ValueError("CLIVER_SYSTEM_ONE_MODEL is required when a System One provider or endpoint is configured.")

    if provider == "laya":
        max_len_raw = values.get("CLIVER_SYSTEM_ONE_MAX_LEN", "").strip()
        provider_obj = LocalSystemOneProvider(
            model=model,
            backend="laya",
            device=values.get("CLIVER_SYSTEM_ONE_DEVICE", "").strip() or "auto",
            max_length=_env_int(values, "CLIVER_SYSTEM_ONE_MAX_LEN", 0, minimum=1) if max_len_raw else None,
        )
        return local_system_one(provider=provider_obj)

    if provider == "bailian":
        workspace_id = values.get("CLIVER_SYSTEM_ONE_WORKSPACE_ID", "").strip()
        if not workspace_id and not base_url:
            raise ValueError("bailian requires CLIVER_SYSTEM_ONE_WORKSPACE_ID (or CLIVER_SYSTEM_ONE_BASE_URL).")
        return presets.bailian_system_one(
            workspace_id=workspace_id,
            api_key=_system_one_api_key(values, "dashscope"),
            model=model,
            region=values.get("CLIVER_SYSTEM_ONE_REGION", "").strip() or "cn-beijing",
            base_url=base_url,
        )

    if provider == "typesafe":
        return presets.typesafe_system_one(
            api_key=_system_one_api_key(values, "typesafe"),
            model=model,
            base_url=base_url,
        )

    if provider:
        raise ValueError(
            f"unknown System One provider {provider!r}: use bailian, typesafe, or laya "
            "(a custom endpoint sets CLIVER_SYSTEM_ONE_BASE_URL and leaves the provider unset)."
        )
    if not base_url:
        raise ValueError(
            "a System One endpoint is required: set CLIVER_SYSTEM_ONE_PROVIDER "
            "(bailian/typesafe/laya) or CLIVER_SYSTEM_ONE_BASE_URL."
        )
    return ModelBinding.system_one(
        model=model,
        api_key=_system_one_api_key(values, None) or "",
        base_url=base_url,
    )


def selector_from_env(
    system_one_binding: ModelBinding | None,
    env: Mapping[str, str] | None = None,
) -> SystemOneSelector | None:
    """Build the candidate selector when ``CLIVER_SELECTOR`` is on.

    Selection activates per category only when the candidate count exceeds
    ``CLIVER_SELECTOR_MIN_TOOLS`` / ``CLIVER_SELECTOR_MIN_SKILLS``.  The
    switch alone is not enough: without a configured System One model the
    selector stays off (with a warning), because the main model receives
    every candidate as before.

    The close read defaults on for hosted decision models and off for a
    local model (laya), whose wide pass ranks more reliably than its
    full-detail answers; ``CLIVER_SELECTOR_CLOSE_READ`` overrides either.
    """
    values = os.environ if env is None else env
    if not _env_bool(values, "CLIVER_SELECTOR", False):
        return None
    if system_one_binding is None:
        print(
            "warning: CLIVER_SELECTOR is on but no System One model is configured; "
            "set CLIVER_SYSTEM_ONE_MODEL (and CLIVER_SYSTEM_ONE_PROVIDER) to enable selection.",
            file=sys.stderr,
        )
        return None

    timeout_raw = values.get("CLIVER_SELECTOR_TIMEOUT", "").strip()
    return SystemOneSelector(
        binding=system_one_binding,
        min_tools=_env_int(values, "CLIVER_SELECTOR_MIN_TOOLS", 20),
        min_skills=_env_int(values, "CLIVER_SELECTOR_MIN_SKILLS", 20),
        top_k_tools=_env_int(values, "CLIVER_SELECTOR_TOP_K_TOOLS", 10, minimum=1),
        top_k_skills=_env_int(values, "CLIVER_SELECTOR_TOP_K_SKILLS", 5, minimum=1),
        chunk_size=_env_int(values, "CLIVER_SELECTOR_CHUNK_SIZE", 20, minimum=1),
        shortlist=_env_int(values, "CLIVER_SELECTOR_SHORTLIST", 12, minimum=1),
        # The close read sharpens hosted models (Bailian/Jev) but a local laya
        # model ranks better from the wide pass: its full-detail Choice is
        # noisy and its Noul gate saturates.  Explicit env still wins.
        close_read=_env_bool(values, "CLIVER_SELECTOR_CLOSE_READ", system_one_binding.protocol != "local"),
        fit_threshold=_env_float(values, "CLIVER_SELECTOR_FIT_THRESHOLD", 0.3),
        timeout=_env_positive_float(values, "CLIVER_SELECTOR_TIMEOUT") if timeout_raw else None,
    )


def build_agent(
    binding: ModelBinding,
    *,
    cwd: Path | None = None,
    config_dir: Path | None = None,
    home: Path | None = None,
    permission_callback=None,
    on_event=None,
    system_one_binding: ModelBinding | None = None,
    candidate_selector=None,
) -> AgentCore:
    """Build an ``AgentCore`` with CLIver's tools, skills, and identity."""
    working_dir = Path.cwd() if cwd is None else Path(cwd)
    mounts: dict[str, ModelBinding] = {"llm": binding}
    if system_one_binding is not None:
        mounts["system_one"] = system_one_binding
    return AgentCore(
        ModelBindings(**mounts),
        builtin_tools=discover_builtin_tools(),
        skill_manager=create_skill_manager(config_dir=config_dir, cwd=working_dir, home=home),
        permission_callback=permission_callback,
        candidate_selector=candidate_selector,
        on_event=on_event,
        builtin_system_prompt=_system_prompt(working_dir),
    )


# ── Environment helpers ─────────────────────────────────────


def _env_int(values: Mapping[str, str], name: str, default: int, *, minimum: int = 0) -> int:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        number = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer (got {raw!r}).") from exc
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {number}).")
    return number


def _env_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number (got {raw!r}).") from exc


def _env_positive_float(values: Mapping[str, str], name: str) -> float | None:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        number = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number (got {raw!r}).") from exc
    if number <= 0:
        raise ValueError(f"{name} must be positive (got {number}).")
    return number


def _env_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (got {raw!r}).")


def _system_one_api_key(values: Mapping[str, str], brand: str | None) -> str | None:
    """Explicit key first, then the provider's conventional variable."""
    explicit = values.get("CLIVER_SYSTEM_ONE_API_KEY")
    if explicit is not None:
        return explicit
    if brand is None:
        return None
    return next(
        (values[name] for name in API_KEY_ENV.get(brand, ()) if values.get(name)),
        None,
    )


def make_permission_callback(*, assume_yes: bool = False):
    """Return the ASK-tool gate used by the one-shot CLI.

    ``--yes`` approves everything.  Otherwise an interactive terminal is
    prompted, and a non-interactive run denies with a hint instead of hanging.
    """

    async def decide(context: ToolCallContext) -> Decision:
        if assume_yes:
            return Decision.allow()
        if not sys.stdin.isatty():
            return Decision.deny("non-interactive session; rerun with --yes to approve tool calls")
        answer = await asyncio.to_thread(input, f"Allow {context.tool_name}? [y/N] ")
        if answer.strip().lower() in {"y", "yes"}:
            return Decision.allow()
        return Decision.deny("user denied the tool call")

    return decide


def make_event_logger(stream: TextIO | None = None):
    """Return an event handler that reports tool activity on stderr."""

    output = sys.stderr if stream is None else stream

    def write(line: str) -> None:
        print(line, file=output, flush=True)

    async def log(event: InferenceEvent | ToolEvent) -> None:
        if isinstance(event, ToolEvent):
            if event.event is ToolEventType.START:
                write(f"[tool] {event.tool_name} ...")
            elif event.event is ToolEventType.END:
                duration = f" ({event.duration_ms:.0f} ms)" if event.duration_ms is not None else ""
                write(f"[tool] {event.tool_name} done{duration}")
            elif event.event is ToolEventType.ERROR:
                write(f"[tool] {event.tool_name} failed: {event.error}")
            elif event.event is ToolEventType.PERMISSION_DENIED:
                write(f"[tool] {event.tool_name} denied: {event.error}")
        elif event.event is InferenceEventType.ERROR:
            error = (event.data or {}).get("error", "unknown error")
            write(f"[llm] error: {error}")

    return log


def _system_prompt(cwd: Path) -> str:
    return (
        "You are CLIver, a terminal AI assistant. Use the available tools to inspect the "
        "workspace and do work accurately; prefer reading files over guessing. When a task "
        "matches an available skill, activate it before proceeding. "
        f"The current working directory is {cwd}."
    )
