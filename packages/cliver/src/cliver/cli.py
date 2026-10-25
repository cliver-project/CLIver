"""One-shot command line entry point for the ``cliver`` application."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cliver.app import (
    build_agent,
    llm_binding_from_env,
    load_environment,
    make_event_logger,
    make_permission_callback,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    try:
        package_version = version("cliver")
    except PackageNotFoundError:  # running from a source tree without install metadata
        package_version = "unknown"

    parser = argparse.ArgumentParser(
        prog="cliver",
        description="CLIver — one-shot agent work from the terminal.",
    )
    parser.add_argument("prompt", nargs="*", help="prompt to run (reads stdin when piped)")
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
    if not prompt:
        parser.error("a prompt is required (or pipe one on stdin)")

    if args.cwd:
        os.chdir(Path(args.cwd).expanduser())

    binding = llm_binding_from_env(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        protocol=args.protocol,
    )
    agent = build_agent(
        binding,
        cwd=Path.cwd(),
        permission_callback=make_permission_callback(assume_yes=args.yes),
        on_event=make_event_logger(),
    )

    response = asyncio.run(_run(agent, prompt, args.skill, args.max_iterations))
    content = response.message.content or ""
    if content:
        print(content)
    return 0


async def _run(agent, prompt: str, skill: str | None, max_iterations: int):
    try:
        if skill:
            return await agent.run_skill(skill, prompt, max_iterations=max_iterations)
        return await agent.chat(prompt, max_iterations=max_iterations)
    finally:
        await agent.aclose()
