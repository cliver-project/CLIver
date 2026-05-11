"""Workflow compiler -- converts Workflow model into LangGraph CompiledStateGraph."""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from cliver.workflow.condition_eval import evaluate_condition
from cliver.workflow.ref_resolver import build_auto_context, resolve_refs
from cliver.workflow.workflow_models import LLMStep, PythonStep, Step, Workflow

logger = logging.getLogger(__name__)


def _save_step_output(outputs_dir: str, step_id: str, text: str, output_format: str = "md") -> Path:
    """Save step text output to the workflow outputs directory."""
    ext_map = {"json": ".json", "text": ".txt", "txt": ".txt", "markdown": ".md", "md": ".md"}
    ext = ext_map.get(output_format, ".md")
    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{step_id}{ext}"
    file_path.write_text(text, encoding="utf-8")
    return file_path


def create_step_logger(outputs_dir: str, step_id: str):
    """Create a tool-event logger that writes JSONL to a log file.

    Returns (log_fn, log_path) where log_fn accepts a ToolEvent.
    """
    import json as _json

    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / f"{step_id}.log"
    fh = open(log_path, "a", encoding="utf-8")

    def log_fn(event):
        entry = {"tool": event.tool_name, "type": event.event_type.value}
        if event.args:
            entry["args"] = {k: str(v)[:200] for k, v in event.args.items()}
        if event.result:
            entry["result"] = event.result[:500]
        if event.error:
            entry["error"] = event.error
        if event.duration_ms is not None:
            entry["duration_ms"] = round(event.duration_ms)
        fh.write(_json.dumps(entry) + "\n")
        fh.flush()

    return log_fn, log_path


def merge_steps(left: Optional[Dict[str, Any]], right: Dict[str, Any]) -> Dict[str, Any]:
    if left is None:
        return dict(right)
    merged = dict(left)
    merged.update(right)
    return merged


class WorkflowState(TypedDict):
    inputs: Dict[str, Any]
    steps: Annotated[Dict[str, Dict[str, Any]], merge_steps]
    workflow_id: str
    thread_id: str
    outputs_dir: str
    error: Optional[str]


_SKIP_SENTINEL = "__skip__"


class WorkflowCompiler:
    """Compiles a Workflow into a LangGraph CompiledStateGraph."""

    def compile(self, workflow: Workflow, checkpointer=None, app_config=None, on_tool_event=None, base_dir="."):
        graph = StateGraph(WorkflowState)
        steps_by_id: Dict[str, Step] = {s.id: s for s in workflow.steps}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for step in workflow.steps:
            for dep in step.depends_on:
                dependents[dep].append(step.id)

        for step in workflow.steps:
            node_fn = self._make_node(step, workflow, app_config, on_tool_event, base_dir)
            graph.add_node(step.id, node_fn)

        roots = [s for s in workflow.steps if not s.depends_on]
        terminals = [s for s in workflow.steps if s.id not in dependents]

        for step in roots:
            graph.add_edge("__start__", step.id)

        for step in terminals:
            graph.add_edge(step.id, END)

        conditional_sources_handled = set()

        for step in workflow.steps:
            if not step.depends_on:
                continue

            for dep_id in step.depends_on:
                if dep_id in conditional_sources_handled:
                    continue

                siblings = dependents.get(dep_id, [])
                conditional_siblings = [sid for sid in siblings if steps_by_id[sid].condition is not None]

                if len(conditional_siblings) > 1:
                    self._add_conditional_edges(graph, dep_id, conditional_siblings, steps_by_id)
                    conditional_sources_handled.add(dep_id)
                elif step.condition is None:
                    try:
                        graph.add_edge(dep_id, step.id)
                    except ValueError:
                        pass

        return graph.compile(checkpointer=checkpointer)

    def _add_conditional_edges(
        self,
        graph: StateGraph,
        source_id: str,
        target_ids: List[str],
        steps_by_id: Dict[str, Step],
    ):
        conditions_map = {}
        for tid in target_ids:
            conditions_map[tid] = steps_by_id[tid].condition

        def router(state: WorkflowState) -> List[str]:
            results = []
            for tid, condition in conditions_map.items():
                if evaluate_condition(condition, state.get("steps", {})):
                    results.append(tid)
            if not results:
                results.append(_SKIP_SENTINEL)
            return results

        destinations = {tid: tid for tid in target_ids}
        destinations[_SKIP_SENTINEL] = END
        graph.add_conditional_edges(source_id, router, destinations)

    def _make_node(self, step: Step, workflow: Workflow, app_config=None, on_tool_event=None, base_dir="."):
        if isinstance(step, LLMStep):
            return self._make_llm_node(step, workflow, app_config, on_tool_event)
        elif isinstance(step, PythonStep):
            return self._make_python_node(step, workflow, base_dir)
        raise ValueError(f"Unknown step type: {type(step)}")

    @staticmethod
    def _make_llm_node(step: LLMStep, workflow: Workflow, app_config=None, on_tool_event=None):
        async def node(state: WorkflowState) -> dict:
            from cliver.workflow.node_runners import run_llm_node

            auto_ctx = build_auto_context(step.depends_on, state)
            prompt = resolve_refs(step.prompt, state)
            if auto_ctx:
                prompt = f"{auto_ctx}\n\n{prompt}"

            start = time.time()
            try:
                result = await run_llm_node(
                    prompt=prompt,
                    agent_name=step.agent,
                    tools=step.tools,
                    output_format=step.output_format,
                    outputs_dir=state["outputs_dir"],
                    step_id=step.id,
                    app_config=app_config,
                    on_tool_event=on_tool_event,
                )
            except Exception as e:
                logger.error("LLM node %s failed: %s", step.id, e)
                result = {"error": str(e)}

            elapsed = time.time() - start
            result["_execution_time"] = round(elapsed, 2)
            return {"steps": {step.id: result}}

        node.__name__ = f"llm_{step.id}"
        return node

    @staticmethod
    def _make_python_node(step: PythonStep, workflow: Workflow, base_dir: str = "."):
        async def node(state: WorkflowState) -> dict:
            from cliver.workflow.node_runners import run_python_node

            node_inputs = dict(state.get("inputs", {}))
            for dep_id in step.depends_on:
                dep_output = state.get("steps", {}).get(dep_id)
                if dep_output:
                    node_inputs[dep_id] = dep_output

            file_path = None
            if step.file:
                file_path = step.file
                if not Path(file_path).is_absolute():
                    file_path = str(Path(base_dir) / file_path)

            step_output_dir = Path(state["outputs_dir"]) / step.id
            step_output_dir.mkdir(parents=True, exist_ok=True)

            start = time.time()
            try:
                result = run_python_node(node_inputs, file_path=file_path, code=step.code, state=dict(state))
            except Exception as e:
                logger.error("Python node %s failed: %s", step.id, e)
                result = {"error": str(e)}

            elapsed = time.time() - start
            result["_execution_time"] = round(elapsed, 2)

            result_text = result.get("result", "")
            if result_text:
                (step_output_dir / "result.txt").write_text(str(result_text), encoding="utf-8")

            return {"steps": {step.id: result}}

        node.__name__ = f"python_{step.id}"
        return node
