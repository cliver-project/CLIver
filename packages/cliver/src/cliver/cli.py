"""Command line entry point for the ``cliver`` application."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cliver_core import ProviderError

from cliver.app import (
    build_agent,
    llm_binding_from_env,
    load_environment,
    make_event_logger,
    make_permission_callback,
    selector_from_env,
    system_one_binding_from_env,
)
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
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="prompt to run (reads stdin when piped; starts the REPL when omitted)",
    )
    parser.add_argument("-m", "--model", help="chat model (default: CLIVER_MODEL)")
    parser.add_argument("--provider", help="known provider brand (default: CLIVER_PROVIDER)")
    parser.add_argument("--base-url", help="endpoint base URL (default: CLIVER_BASE_URL)")
    parser.add_argument("--protocol", choices=("openai", "anthropic"), help="wire protocol")
    parser.add_argument("--skill", help="run this skill with the prompt")
    parser.add_argument("--cwd", help="working directory and project skill root")
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

        binding = llm_binding_from_env(
            model=args.model,
            provider=args.provider,
            base_url=args.base_url,
            protocol=args.protocol,
        )
        system_one_binding = system_one_binding_from_env()
        agent = build_agent(
            binding,
            cwd=Path.cwd(),
            permission_callback=make_permission_callback(assume_yes=args.yes),
            on_event=make_event_logger(),
            system_one_binding=system_one_binding,
            candidate_selector=selector_from_env(system_one_binding),
        )
    except (OSError, ImportError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    try:
        if interactive:
            return asyncio.run(
                _run_repl(agent, binding, args.max_iterations, system_one_binding),
            )

        response = asyncio.run(_run(agent, prompt, args.skill, args.max_iterations, system_one_binding))
    except ProviderError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        hint = _endpoint_hint(binding)
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


async def _run(agent, prompt: str, skill: str | None, max_iterations: int, system_one_binding=None):
    try:
        if skill:
            return await agent.run_skill(skill, prompt, max_iterations=max_iterations)
        return await agent.chat(prompt, max_iterations=max_iterations)
    finally:
        await _close_agent(agent, system_one_binding)


async def _run_repl(agent, binding, max_iterations: int, system_one_binding=None) -> int:
    try:
        return await run_repl(
            agent,
            max_iterations=max_iterations,
            model=getattr(binding, "model", ""),
            endpoint=_endpoint_hint(binding),
        )
    finally:
        await _close_agent(agent, system_one_binding)


def _endpoint_hint(binding) -> str:
    base_url = getattr(binding, "base_url", "") or ""
    protocol = getattr(binding, "protocol", "") or ""
    return f"{base_url} ({protocol} protocol)" if base_url else ""


async def _close_agent(agent, system_one_binding=None) -> None:
    await agent.aclose()
    # A local System One model (laya) is caller-supplied, so the binding
    # never closes it — the CLI owns that provider and releases it here.
    provider = None if system_one_binding is None else system_one_binding.provider
    if provider is not None and hasattr(provider, "close"):
        await provider.close()
