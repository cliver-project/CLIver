"""Built-in tool for running multiple LLM tasks in parallel.

The LLM uses this tool to execute multiple independent prompts concurrently,
combining results when all complete. Lightweight alternative to subagent delegation.
"""

import asyncio
import logging
import time
from typing import List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_agent_core

logger = logging.getLogger(__name__)


class ParallelTasksInput(BaseModel):
    """Input schema for parallel_tasks."""

    tasks: List[str] = Field(description="List of prompts to execute in parallel. Each prompt is an independent task.")


class ParallelTasksTool(BaseTool):
    """Run multiple independent tasks in parallel and combine results."""

    name: str = "Parallel"
    description: str = (
        "Execute multiple independent prompts in parallel and return all results. "
        "Use when the user asks for multiple unrelated things that can be done concurrently, "
        "such as 'research X and also check Y' or 'summarize these 3 documents'. "
        "Each task runs independently with its own context. "
        "Do NOT use this for tasks that depend on each other's results."
    )
    args_schema: Type[BaseModel] = ParallelTasksInput

    def _run(self, tasks: List[str]) -> str:
        executor = get_agent_core()
        if executor is None:
            return "Error: parallel_tasks is not available in this session."

        if not tasks:
            return "No tasks provided."

        if len(tasks) == 1:
            return "Only one task provided — use a regular prompt instead of parallel_tasks."

        try:
            results = asyncio.run(self._run_parallel(executor, tasks))
        except RuntimeError:
            # Event loop already running (interactive mode)
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(self._run_parallel(executor, tasks))

        return self._format_results(tasks, results)

    @staticmethod
    async def _run_parallel(executor, tasks: List[str]) -> List[dict]:
        """Run all tasks concurrently and collect results."""
        start = time.monotonic()

        async def run_one(prompt: str, index: int) -> dict:
            task_start = time.monotonic()
            try:
                response = await executor.process_user_input(user_input=prompt)
                from cliver.media_handler import extract_response_text

                text = extract_response_text(response, fallback="(empty response)")
                return {
                    "index": index,
                    "prompt": prompt,
                    "result": text,
                    "success": True,
                    "duration": round(time.monotonic() - task_start, 1),
                }
            except Exception as e:
                return {
                    "index": index,
                    "prompt": prompt,
                    "result": f"Error: {e}",
                    "success": False,
                    "duration": round(time.monotonic() - task_start, 1),
                }

        results = await asyncio.gather(*[run_one(t, i) for i, t in enumerate(tasks)])
        total = round(time.monotonic() - start, 1)
        logger.info("parallel_tasks completed %d tasks in %.1fs", len(tasks), total)
        return sorted(results, key=lambda r: r["index"])

    @staticmethod
    def _format_results(tasks: List[str], results: List[dict]) -> str:
        """Format parallel results into a readable summary."""
        lines = [f"Completed {len(results)} tasks in parallel:\n"]
        for r in results:
            status = "OK" if r["success"] else "FAILED"
            lines.append(f"--- Task {r['index'] + 1} [{status}, {r['duration']}s] ---")
            lines.append(f"Prompt: {r['prompt'][:100]}")
            lines.append(f"Result: {r['result']}")
            lines.append("")
        return "\n".join(lines)


parallel_tasks = ParallelTasksTool()
