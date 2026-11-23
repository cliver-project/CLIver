"""Review a small repository read-only and return a validated report.

Run it with ``DEEPSEEK_API_KEY`` set:

    python samples/blog/code_review/code_review.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from cliver_core import AgentCore, ExecutionPolicy, ModelBinding, ModelBindings, StructuredResponse, presets
from cliver_core.policy import pinned_cwd, scope_violation
from cliver_core.provider import Provider
from cliver_core.tool import tool
from pydantic import BaseModel, Field

FIXTURE = Path(__file__).resolve().parent / "fixture"

PROMPT = (
    "Review the Python files in this repository for correctness problems. "
    "List the files first, then read each one. Report every concrete problem "
    "with its file, 1-based line number, severity, and a short message."
)


# region: schema
class Finding(BaseModel):
    file: str = Field(description="Path relative to the repository root")
    line: int = Field(description="1-based line number")
    severity: str = Field(description="low, medium, or high")
    message: str = Field(description="What is wrong and why it matters")


class ReviewReport(BaseModel):
    summary: str = Field(description="One short paragraph on the overall state")
    findings: list[Finding] = Field(description="Concrete problems, most severe first")


# endregion: schema


# region: tool
@tool(
    name="InspectRepo",
    description="List the Python files in the review repository, or read one file by path. Read-only.",
    safety="safe",
)
def inspect_repo(path: str = "") -> list[dict]:
    """List the repository's Python files, or read one file with line numbers.

    Paths resolve against the pinned working directory. The ambient
    execution policy refuses anything outside the review repository.
    """
    base = pinned_cwd() or Path.cwd()
    target = (base / path).resolve() if path else base.resolve()
    violation = scope_violation(target, write=False)
    if violation:
        return [{"error": violation}]
    if target.is_dir():
        files = sorted(file.relative_to(target).as_posix() for file in target.rglob("*.py"))
        return [{"text": "\n".join(files) if files else "(no Python files)"}]
    if not target.is_file():
        return [{"error": f"File not found: {path or base}"}]
    label = target.relative_to(base).as_posix() if target.is_relative_to(base) else target.as_posix()
    numbered = "\n".join(
        f"{number:>4}\t{line}" for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1)
    )
    return [{"text": f"### {label}\n{numbered}"}]


# endregion: tool


# region: policy
def review_policy(root: Path) -> ExecutionPolicy:
    """Pin the run to *root*: reads stay inside it and writes go nowhere."""
    return ExecutionPolicy(cwd=root, read_paths=[root], write_paths=[])


# endregion: policy


# region: review
async def review_repo(agent: AgentCore, root: Path = FIXTURE) -> StructuredResponse:
    return await agent.chat_structured(
        PROMPT,
        schema=ReviewReport,
        execution_policy=review_policy(root),
    )


# endregion: review


# region: agent
def build_agent(provider: Provider | None = None) -> AgentCore:
    bindings = (
        ModelBindings(llm=presets.deepseek())
        if provider is None
        else ModelBindings(llm=ModelBinding.llm(model="scripted", provider=provider))
    )
    return AgentCore(bindings, builtin_tools=[inspect_repo])


# endregion: agent


def print_report(report: ReviewReport) -> None:
    print(report.summary)
    for finding in report.findings:
        print(f"{finding.file}:{finding.line}: {finding.severity}: {finding.message}")


async def main() -> None:
    agent = build_agent()
    try:
        result = await review_repo(agent)
    finally:
        await agent.aclose()

    report = result.value
    assert isinstance(report, ReviewReport)
    print_report(report)
    print(f"attempts: {result.attempts}")


if __name__ == "__main__":
    asyncio.run(main())
