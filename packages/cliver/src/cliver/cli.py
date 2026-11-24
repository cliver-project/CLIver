"""Command line entry point for the ``cliver`` application."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cliver_core import ModelBinding, ModelBindings, ProviderError

from cliver.app import (
    build_agent,
    llm_binding_from_env,
    load_environment,
    make_event_logger,
    make_permission_callback,
    selector_from_env,
    system_one_binding_from_env,
)
from cliver.config import CLIConfig, MCPSelection, create_mcp_client, load_config
from cliver.repl import run_repl

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    try:
        package_version = version("cliver")
    except PackageNotFoundError:  # running from a source tree without install metadata
        package_version = "unknown"

    parser = argparse.ArgumentParser(
        prog="cliver",
        description="CLIver — agent work from the terminal.",
        epilog=(
            "environment:\n"
            "  CLIVER_MODEL      chat model (with --model)\n"
            "  CLIVER_PROVIDER   known provider brand (with --provider)\n"
            "  CLIVER_BASE_URL   endpoint for any compatible server (with --base-url)\n"
            "  CLIVER_PROTOCOL   openai (default) or anthropic (with --protocol)\n"
            "  CLIVER_API_KEY    API key; otherwise the brand's variable (DEEPSEEK_API_KEY, ...)\n"
            "  CLIVER_CONF_DIR   config directory; {config_dir}/.env is loaded at startup\n"
            "  CLIVER_CONFIG     explicit JSON config file (default: user + project config.json)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="prompt to run (reads stdin when piped; starts the REPL when omitted)",
    )
    parser.add_argument("-m", "--model", help="chat model (default: CLIVER_MODEL)")
    parser.add_argument("--provider", help="known provider brand (default: CLIVER_PROVIDER)")
    parser.add_argument("--base-url", help="endpoint base URL (default: CLIVER_BASE_URL)")
    parser.add_argument(
        "--protocol",
        choices=("openai", "anthropic"),
        help="wire protocol (default: CLIVER_PROTOCOL or openai)",
    )
    parser.add_argument("--skill", help="run this skill with the prompt")
    parser.add_argument("--cwd", help="working directory and project skill root")
    parser.add_argument(
        "--config",
        help="JSON configuration file (default: {config_dir}/config.json plus <cwd>/.cliver/config.json)",
    )
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="approve ASK tools without prompting (non-interactive runs need this)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {package_version}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_environment()

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    interactive = not prompt and sys.stdin.isatty() and sys.stdout.isatty()
    if not prompt and not interactive:
        parser.error("a prompt is required (or pipe one on stdin)")
    if not prompt and args.skill:
        parser.error("--skill requires a prompt")

    # Configuration problems are user errors, not crashes: report them as a
    # one-line CLI error (exit 2) instead of a traceback.
    try:
        if args.cwd:
            os.chdir(Path(args.cwd).expanduser())

        config = load_config(config_path=args.config, cwd=Path.cwd())
        run_binding, model_bindings, system_one_binding, _ = _resolve_bindings(config, args)
        display_binding = run_binding.llm if isinstance(run_binding, ModelBindings) else run_binding
        _preflight(run_binding, model_bindings)
        mcp = create_mcp_client(config, cwd=Path.cwd())
        _report_mcp(mcp)
        if config.selector_env:
            candidate_selector = selector_from_env(
                system_one_binding,
                env={**config.selector_env, **os.environ},
            )
        else:
            candidate_selector = selector_from_env(system_one_binding)
        agent = build_agent(
            model_bindings if model_bindings is not None else run_binding,
            cwd=Path.cwd(),
            permission_callback=make_permission_callback(assume_yes=args.yes),
            on_event=make_event_logger(),
            system_one_binding=None if model_bindings is not None else system_one_binding,
            candidate_selector=candidate_selector,
            mcp_client=mcp.client,
            skill_sources=config.skill_sources,
        )
    except (OSError, ImportError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    try:
        if interactive:
            return asyncio.run(
                _run_repl(agent, display_binding, args.max_iterations, system_one_binding, mcp.client),
            )

        response = asyncio.run(
            _run(agent, prompt, args.skill, args.max_iterations, system_one_binding, mcp.client),
        )
    except ProviderError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        hint = _endpoint_hint(display_binding)
        if hint:
            print(f"{parser.prog}: endpoint: {hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"{parser.prog}: interrupted", file=sys.stderr)
        return 130

    content = response.message.content or ""
    if content:
        print(content)
    return 0


_MEDIA_PURPOSES = ("embed", "rerank", "image", "audio", "video", "tts", "asr")


def _resolve_bindings(
    config: CLIConfig,
    args,
) -> tuple[ModelBinding | ModelBindings, ModelBindings | None, ModelBinding | None, list[ModelBinding]]:
    """Combine configuration models with the existing environment overrides.

    Flags and ``CLIVER_*`` variables keep their precedence: an explicit LLM
    setting replaces the configured LLM list, and any ``CLIVER_SYSTEM_ONE_*``
    variable replaces the configured System One list.  Media purposes come
    from the configuration only.

    A single chat binding (with at most one System One binding and no media
    purposes) keeps the legacy ``build_agent(binding, ...)`` shape; anything
    larger becomes a ``ModelBindings``.
    """
    explicit_llm = bool(args.model or args.provider or args.base_url or args.protocol or os.environ.get("CLIVER_MODEL"))
    if explicit_llm:
        llm_bindings = [
            llm_binding_from_env(
                model=args.model,
                provider=args.provider,
                base_url=args.base_url,
                protocol=args.protocol,
            )
        ]
    else:
        llm_bindings = list(config.models.get("llm") or ())
        if not llm_bindings:
            llm_bindings = [
                llm_binding_from_env(
                    model=args.model,
                    provider=args.provider,
                    base_url=args.base_url,
                    protocol=args.protocol,
                )
            ]

    configured = system_one_binding_from_env()
    if configured is not None:
        system_one_bindings = [configured]
    else:
        system_one_bindings = list(config.models.get("system_one") or ())

    system_one = system_one_bindings[0] if system_one_bindings else None
    media = {purpose: config.models[purpose] for purpose in _MEDIA_PURPOSES if config.models.get(purpose)}

    if len(llm_bindings) == 1 and not media and len(system_one_bindings) <= 1:
        return llm_bindings[0], None, system_one, system_one_bindings

    mounts: dict[str, object] = {"llm": llm_bindings, **media}
    if system_one_bindings:
        mounts["system_one"] = system_one_bindings
    model_bindings = ModelBindings(**mounts)
    return model_bindings, model_bindings, system_one, system_one_bindings


def _preflight(binding, model_bindings: ModelBindings | None) -> None:
    """Build configured providers now, so bad settings fail as a one-line error."""
    if model_bindings is None:
        targets = [binding]
    else:
        targets = [*model_bindings.get_all("llm"), *model_bindings.get_all("system_one")]
    for target in targets:
        build = getattr(target, "build_provider", None)
        if build is not None:
            build()


def _report_mcp(selection: MCPSelection) -> None:
    for problem in selection.problems:
        if problem.severity == "info":
            continue
        location = f" ({problem.path})" if problem.path else ""
        print(f"cliver: mcp: {problem.message}{location}", file=sys.stderr)
    if selection.client is not None:
        print(f"cliver: mcp: {selection.summary}", file=sys.stderr)


async def _run(
    agent,
    prompt: str,
    skill: str | None,
    max_iterations: int,
    system_one_binding=None,
    mcp_client=None,
):
    try:
        if skill:
            return await agent.run_skill(skill, prompt, max_iterations=max_iterations)
        return await agent.chat(prompt, max_iterations=max_iterations)
    finally:
        await _close_agent(agent, system_one_binding, mcp_client)


async def _run_repl(
    agent,
    binding,
    max_iterations: int,
    system_one_binding=None,
    mcp_client=None,
) -> int:
    try:
        return await run_repl(
            agent,
            max_iterations=max_iterations,
            model=getattr(binding, "model", ""),
            endpoint=_endpoint_hint(binding),
        )
    finally:
        await _close_agent(agent, system_one_binding, mcp_client)


def _endpoint_hint(binding) -> str:
    base_url = getattr(binding, "base_url", "") or ""
    protocol = getattr(binding, "protocol", "") or ""
    return f"{base_url} ({protocol} protocol)" if base_url else ""


async def _close_agent(agent, system_one_binding=None, mcp_client=None) -> None:
    try:
        await agent.aclose()
    finally:
        # A local System One model (laya) is caller-supplied, so the binding
        # never closes it — the CLI owns that provider and releases it here.
        # The explicit primary covers the legacy single-binding shape; the
        # bindings cover a configured System One list.  De-duplicate by
        # provider identity so a primary is closed once.
        providers = []
        if system_one_binding is not None:
            providers.append(system_one_binding)
        bindings = getattr(agent, "bindings", None)
        if bindings is not None and hasattr(bindings, "get_all"):
            providers.extend(bindings.get_all("system_one"))
        seen: set[int] = set()
        for item in providers:
            provider = getattr(item, "provider", None)
            if provider is None or id(provider) in seen:
                continue
            seen.add(id(provider))
            close = getattr(provider, "close", None)
            if close is not None:
                await close()
        if mcp_client is not None:
            await mcp_client.close()
