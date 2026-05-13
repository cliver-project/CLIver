"""CellExecutor — type-specific cell execution."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict

from cliver.notebook.ref_resolver import resolve_refs

if TYPE_CHECKING:
    from cliver.notebook.models import Cell

logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when LLM cell output fails verification after all retries."""

    pass


class CellExecutor:
    """Dispatches cell execution by type."""

    async def execute(self, cell: "Cell", runtime) -> Dict[str, Any]:
        handlers = {
            "config": self._execute_config,
            "llm": self._execute_llm,
            "code": self._execute_code,
            "display": self._execute_display,
        }
        handler = handlers.get(cell.type)
        if not handler:
            raise ValueError(f"Unknown cell type: '{cell.type}'")
        return await handler(cell, runtime)

    async def _execute_config(self, cell: "Cell", runtime) -> Dict[str, Any]:
        return dict(cell.outputs)

    async def _execute_llm(self, cell: "Cell", runtime) -> Dict[str, Any]:
        prompt = resolve_refs(cell.inputs.get("prompt", ""), runtime.variables)
        agent_name = cell.inputs.get("agent", "")
        if agent_name and "${" in agent_name:
            agent_name = resolve_refs(agent_name, runtime.variables)
        agent_name = agent_name or getattr(runtime.notebook, "default_agent", None)

        agent = runtime.agent_factory.create(agent_name or None)
        ctx = getattr(runtime.notebook, "context", {})
        working_dir = ctx.get("working_dir") if isinstance(ctx, dict) else None
        await agent.initialize({"working_dir": working_dir})

        verification = cell.inputs.get("verification")
        if not verification or not verification.get("expected"):
            # No verification — single execution (existing behavior)
            result = await agent.run(prompt)
            return self._build_llm_outputs(result, cell)

        # Verification loop
        expected = verification["expected"]
        max_retries = verification.get("max_retries", 3)
        timeout_s = verification.get("timeout_s", 300)
        verifier_agent_name = verification.get("verifier_agent") or agent_name

        verifier = runtime.agent_factory.create(verifier_agent_name or None)
        await verifier.initialize({"working_dir": working_dir})

        original_prompt = prompt
        last_reason = ""

        for attempt in range(1, max_retries + 1):
            try:
                result = await asyncio.wait_for(agent.run(prompt), timeout=timeout_s)
            except asyncio.TimeoutError:
                raise VerificationError(
                    f"LLM execution timed out after {timeout_s}s (attempt {attempt}/{max_retries})"
                ) from None

            outputs = self._build_llm_outputs(result, cell)

            # Run verifier
            verify_prompt = (
                f"Verify if the following output matches the expected result.\n\n"
                f"Output:\n{result.text}\n\n"
                f"Expected:\n{expected}\n\n"
                f"Respond ONLY with a JSON object: "
                f'{{"pass": true, "reason": "explanation"}} or '
                f'{{"pass": false, "reason": "explanation"}}'
            )

            try:
                verify_result = await asyncio.wait_for(verifier.run(verify_prompt), timeout=60)
                verdict = self._parse_verdict(verify_result.text)
            except asyncio.TimeoutError:
                verdict = {"pass": False, "reason": "Verification timed out"}
            except Exception as e:
                verdict = {"pass": False, "reason": f"Verification error: {e}"}

            if verdict.get("pass"):
                outputs["_verification"] = {
                    "passed": True,
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "reason": verdict.get("reason", "Verification passed"),
                }
                return outputs

            last_reason = verdict.get("reason", "Output did not match expectations")
            logger.info(
                "Cell '%s' verification failed (attempt %d/%d): %s",
                cell.id,
                attempt,
                max_retries,
                last_reason,
            )

            # Retry with feedback
            prompt = (
                f"{original_prompt}\n\n"
                f"[Previous attempt did not meet expectations: {last_reason}. "
                f"Please try again and ensure the output matches: {expected}]"
            )

        raise VerificationError(f"Verification failed after {max_retries} attempts. Last reason: {last_reason}")

    def _build_llm_outputs(self, result, cell: "Cell") -> Dict[str, Any]:
        """Build outputs dict from an AgentResult."""
        outputs: Dict[str, Any] = {"text": result.text}
        if result.artifacts:
            outputs["artifacts"] = [
                {"path": a.path, "media_type": a.media_type, "size": a.size} for a in result.artifacts
            ]
        if cell.inputs.get("output_format") == "json":
            try:
                outputs["data"] = json.loads(result.text)
            except (json.JSONDecodeError, TypeError):
                pass
        return outputs

    @staticmethod
    def _parse_verdict(text: str) -> dict:
        """Parse verifier response into {pass: bool, reason: str}."""
        try:
            # Try to extract JSON from the response
            text = text.strip()
            # Handle markdown code blocks
            if "```" in text:
                import re

                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)
            data = json.loads(text)
            return {
                "pass": bool(data.get("pass", False)),
                "reason": str(data.get("reason", "")),
            }
        except (json.JSONDecodeError, TypeError):
            # If we can't parse, check for keywords
            lower = text.lower()
            if "pass" in lower and "true" in lower:
                return {"pass": True, "reason": text[:200]}
            return {"pass": False, "reason": f"Could not parse verifier response: {text[:200]}"}

    async def _execute_code(self, cell: "Cell", runtime) -> Dict[str, Any]:
        source = cell.inputs.get("source", "")
        if not source.strip():
            raise ValueError("Code cell has no source code")

        namespace: dict = {}
        compiled = compile(source, f"<cell:{cell.id}>", "exec")
        exec(compiled, namespace)  # noqa: S102

        run_fn = namespace.get("run")
        if not run_fn or not callable(run_fn):
            raise ValueError("Code cell must define a callable run(ctx) function")

        timeout = cell.inputs.get("timeout_s", 300)
        result = await asyncio.wait_for(
            asyncio.to_thread(run_fn, runtime.ctx),
            timeout=timeout,
        )

        if not isinstance(result, dict):
            raise TypeError(f"run() must return dict, got {type(result).__name__}")

        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            raise TypeError(f"run() returned non-JSON-serializable value: {e}") from e

        return result

    async def _execute_display(self, cell: "Cell", runtime) -> Dict[str, Any]:
        content = cell.inputs.get("content", "")
        if content and "${" in content:
            resolve_refs(content, runtime.variables)
        return {}
