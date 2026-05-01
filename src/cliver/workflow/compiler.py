"""
WorkflowCompiler — translates CLIver YAML Workflow definitions into LangGraph StateGraphs.
"""

import logging
import time
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from cliver.workflow.context_renderer import evaluate_condition, render_template
from cliver.workflow.workflow_models import (
    AgentConfig,
    DecisionStep,
    ExecutionContext,
    FunctionStep,
    HumanStep,
    LLMStep,
    Workflow,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


def merge_steps(left: Optional[Dict[str, Any]], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer for WorkflowState.steps — merges new step results into existing."""
    if left is None:
        return dict(right)
    merged = dict(left)
    merged.update(right)
    return merged


class WorkflowState(TypedDict):
    """LangGraph state schema for workflow execution."""

    inputs: Dict[str, Any]
    steps: Annotated[Dict[str, Dict[str, Any]], merge_steps]
    outputs_dir: str
    workflow_name: str
    execution_id: str
    error: Optional[str]


def _state_to_execution_context(state: dict) -> ExecutionContext:
    """Convert LangGraph state to ExecutionContext for Jinja2 rendering."""
    return ExecutionContext(
        workflow_name=state.get("workflow_name", ""),
        execution_id=state.get("execution_id"),
        inputs=state.get("inputs", {}),
        steps=state.get("steps", {}),
    )


def _save_step_output(outputs_dir: str, step_id: str, content: str, fmt: str) -> None:
    """Save step output to a file in the run directory."""
    if not outputs_dir:
        return
    dir_path = Path(outputs_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    ext_map = {"md": "md", "json": "json", "txt": "txt", "yaml": "yaml", "code": "txt"}
    filename = f"{step_id}.{ext_map.get(fmt, 'md')}"
    (dir_path / filename).write_text(content, encoding="utf-8")
    logger.info("Saved step output: %s/%s", outputs_dir, filename)


def _load_overview(workflow: Workflow) -> Optional[str]:
    """Load the workflow overview from inline text or file."""
    if workflow.overview:
        return workflow.overview
    if workflow.overview_file:
        try:
            return Path(workflow.overview_file).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Could not read overview file %s: %s", workflow.overview_file, e)
    return None


class WorkflowCompiler:
    """Compiles a CLIver Workflow YAML model into a LangGraph CompiledStateGraph."""

    def compile(self, workflow: Workflow, checkpointer=None, store=None, base_dir=None):
        """Compile a workflow definition into an executable LangGraph graph.

        Args:
            workflow: The workflow definition to compile.
            checkpointer: LangGraph checkpointer for persistence.
            store: WorkflowStore for resolving child workflows by name.
            base_dir: Base directory for resolving relative workflow_file paths.
        """
        builder = StateGraph(WorkflowState)
        agents = workflow.agents or {}
        overview = _load_overview(workflow)
        steps_by_id = {step.id: step for step in workflow.steps}

        # Add nodes
        for step in workflow.steps:
            if isinstance(step, DecisionStep):
                builder.add_node(step.id, self._make_passthrough_node(step))
            elif isinstance(step, LLMStep):
                agent_config = agents.get(step.agent) if step.agent else None
                builder.add_node(step.id, self._make_llm_node(step, agent_config, overview))
            elif isinstance(step, HumanStep):
                builder.add_node(step.id, self._make_human_node(step))
            elif isinstance(step, FunctionStep):
                builder.add_node(step.id, self._make_function_node(step))
            elif isinstance(step, WorkflowStep):
                child_wf, child_base = self._resolve_child_workflow(step, store, base_dir)
                child_graph = self.compile(child_wf, checkpointer=checkpointer, store=store, base_dir=child_base)
                builder.add_node(step.id, self._make_workflow_node(step, child_wf, child_graph))

        # Add edges based on depends_on
        for step in workflow.steps:
            if not step.depends_on:
                builder.add_edge("__start__", step.id)
            else:
                for dep_id in step.depends_on:
                    dep_step = steps_by_id.get(dep_id)
                    if isinstance(dep_step, DecisionStep):
                        pass  # handled by conditional edges
                    else:
                        builder.add_edge(dep_id, step.id)

        # Add conditional edges for DecisionSteps
        for step in workflow.steps:
            if isinstance(step, DecisionStep):
                targets = {b.next_step for b in step.branches}
                if step.default:
                    targets.add(step.default)
                router = self._make_decision_router(step)
                builder.add_conditional_edges(step.id, router, {t: t for t in targets})

        # Connect terminal nodes to END
        has_dependents = set()
        for step in workflow.steps:
            for dep in step.depends_on:
                has_dependents.add(dep)
        for step in workflow.steps:
            if isinstance(step, DecisionStep):
                has_dependents.add(step.id)

        for step in workflow.steps:
            if step.id not in has_dependents and not isinstance(step, DecisionStep):
                builder.add_edge(step.id, END)

        return builder.compile(checkpointer=checkpointer)

    @staticmethod
    def _resolve_child_workflow(step: WorkflowStep, store, base_dir):
        """Load and validate a child workflow at compile time.

        Returns (child_workflow, child_base_dir).
        """
        from cliver.workflow.persistence import WorkflowStore as _WFStore

        child_wf = None
        child_base = base_dir

        if step.workflow and store:
            child_wf = store.load_workflow(step.workflow)
            if child_wf:
                child_base = str(store.workflows_dir)

        if not child_wf and step.workflow_file:
            file_path = Path(step.workflow_file)
            if not file_path.is_absolute() and base_dir:
                file_path = Path(base_dir) / file_path
            child_wf = _WFStore.load_workflow_from_file(file_path)
            if child_wf:
                child_base = str(file_path.parent.resolve())

        if not child_wf:
            tried = []
            if step.workflow:
                tried.append(f"name '{step.workflow}'")
            if step.workflow_file:
                tried.append(f"file '{step.workflow_file}'")
            raise ValueError(f"Workflow not found at compile time (tried {', '.join(tried)})")

        return child_wf, child_base

    @staticmethod
    def _make_passthrough_node(step: DecisionStep):
        async def node(state):
            return {"steps": {step.id: {"outputs": {"status": "decided"}, "status": "completed"}}}

        return node

    @staticmethod
    def _make_llm_node(step: LLMStep, agent_config: Optional[AgentConfig], overview: Optional[str]):
        async def node(state, config):
            factory = config["configurable"].get("subagent_factory")
            context = _state_to_execution_context(state)
            rendered_prompt = render_template(step.prompt, context)

            if factory:
                effective_config = agent_config or AgentConfig(model=step.model)
                subagent = factory.create(effective_config)
            else:
                subagent = config["configurable"]["agent_core"]

            def system_appender():
                parts = []
                if overview:
                    parts.append(f"# Workflow Overview\n\n{overview}")
                if agent_config and agent_config.system_message:
                    parts.append(agent_config.system_message)
                return "\n\n".join(parts) if parts else ""

            effective_model = (agent_config.model if agent_config else None) or step.model

            start = time.time()
            try:
                response = await subagent.process_user_input(
                    user_input=rendered_prompt,
                    model=effective_model,
                    system_message_appender=system_appender,
                    images=render_template(step.images, context) if step.images else None,
                    audio_files=render_template(step.audio_files, context) if step.audio_files else None,
                    video_files=render_template(step.video_files, context) if step.video_files else None,
                    files=render_template(step.files, context) if step.files else None,
                )
                from cliver.media_handler import extract_response_text

                result_text = extract_response_text(response)
                _save_step_output(state["outputs_dir"], step.id, result_text, step.output_format)

                outputs = {"result": result_text}
                if step.outputs:
                    for name in step.outputs:
                        outputs[name] = result_text

                return {
                    "steps": {
                        step.id: {
                            "outputs": outputs,
                            "status": "completed",
                            "execution_time": time.time() - start,
                        }
                    }
                }
            except Exception as e:
                logger.error("LLM step '%s' failed: %s", step.id, e)
                return {
                    "steps": {
                        step.id: {
                            "outputs": {"error": str(e)},
                            "status": "failed",
                            "execution_time": time.time() - start,
                        }
                    },
                    "error": f"Step '{step.id}' failed: {e}",
                }

        return node

    @staticmethod
    def _make_human_node(step: HumanStep):
        async def node(state, config):
            context = _state_to_execution_context(state)
            rendered_prompt = render_template(step.prompt, context)

            if step.auto_confirm:
                result = "auto-confirmed"
            else:
                result = interrupt({"prompt": rendered_prompt, "step_id": step.id})

            return {"steps": {step.id: {"outputs": {"result": result}, "status": "completed"}}}

        return node

    @staticmethod
    def _make_function_node(step: FunctionStep):
        async def node(state, config):
            import importlib

            context = _state_to_execution_context(state)
            start = time.time()

            try:
                module_path, func_name = step.function.rsplit(".", 1)
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)

                rendered_inputs = render_template(step.inputs, context) if step.inputs else {}
                result = func(**rendered_inputs) if rendered_inputs else func()

                outputs = {"result": result}
                if step.outputs and isinstance(result, dict):
                    for name in step.outputs:
                        if name in result:
                            outputs[name] = result[name]

                return {
                    "steps": {
                        step.id: {
                            "outputs": outputs,
                            "status": "completed",
                            "execution_time": time.time() - start,
                        }
                    }
                }
            except Exception as e:
                return {
                    "steps": {
                        step.id: {
                            "outputs": {"error": str(e)},
                            "status": "failed",
                            "execution_time": time.time() - start,
                        }
                    },
                    "error": f"Function step '{step.id}' failed: {e}",
                }

        return node

    @staticmethod
    def _make_workflow_node(step: WorkflowStep, child_workflow: Workflow, child_graph):
        """Create a node that invokes a pre-compiled child workflow graph.

        The child graph shares the parent's checkpointer (passed at compile time)
        and gets a derived thread_id for its own checkpoint chain.
        """

        async def node(state, config):
            context = _state_to_execution_context(state)
            start = time.time()

            try:
                rendered_inputs = render_template(step.workflow_inputs, context) if step.workflow_inputs else None

                child_state = {
                    "inputs": child_workflow.get_initial_inputs(rendered_inputs),
                    "steps": {},
                    "outputs_dir": str(Path(state["outputs_dir"]) / step.id),
                    "workflow_name": child_workflow.name,
                    "execution_id": state["execution_id"],
                    "error": None,
                }

                parent_thread = config["configurable"]["thread_id"]
                child_config = {
                    "configurable": {
                        "thread_id": f"{parent_thread}:{step.id}",
                        "agent_core": config["configurable"].get("agent_core"),
                        "subagent_factory": config["configurable"].get("subagent_factory"),
                        "workflow_store": config["configurable"].get("workflow_store"),
                        "db_path": config["configurable"].get("db_path"),
                        "app_config": config["configurable"].get("app_config"),
                        "skill_manager": config["configurable"].get("skill_manager"),
                        "workflow_base_dir": config["configurable"].get("workflow_base_dir"),
                    }
                }

                sub_result = await child_graph.ainvoke(child_state, child_config)

                return {
                    "steps": {
                        step.id: {
                            "outputs": {"result": sub_result},
                            "status": "completed",
                            "execution_time": time.time() - start,
                        }
                    }
                }
            except Exception as e:
                return {
                    "steps": {
                        step.id: {
                            "outputs": {"error": str(e)},
                            "status": "failed",
                            "execution_time": time.time() - start,
                        }
                    },
                    "error": f"Workflow step '{step.id}' failed: {e}",
                }

        return node

    @staticmethod
    def _make_decision_router(step: DecisionStep):
        def route(state) -> str:
            context = _state_to_execution_context(state)
            for branch in step.branches:
                if evaluate_condition(branch.condition, context):
                    return branch.next_step
            return step.default or END

        return route
