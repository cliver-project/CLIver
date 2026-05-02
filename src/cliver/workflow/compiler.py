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

# Tools excluded from workflow step execution — each step is a focused
# execution unit, not a planning session.
_WORKFLOW_EXCLUDED_TOOLS = frozenset(
    {
        "TodoWrite",
        "TodoRead",
        "Skill",
        "CliverHelp",
        "CreateTask",
        "WorkflowValidate",
        "SearchSessions",
        "Ask",
    }
)

_WORKFLOW_STEP_SYSTEM_INSTRUCTION = (
    "You are executing a workflow step autonomously. "
    "Do NOT ask for user confirmation or clarification. "
    "Make the best decision based on the information available and proceed. "
    "Do NOT use the Ask tool. Complete the task directly.\n\n"
    "ALL output files (text, images, audio, video, code, data) MUST be saved "
    "to the designated outputs directory provided below. Do NOT use any other directory. "
    "Reference files from prior steps using the paths listed under Available Files."
)


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


def _save_step_media(outputs_dir: str, step_id: str, response, agent_core, model: str) -> list[str]:
    """Extract and save media files from an LLM response using MultimediaResponseHandler.

    Returns list of saved file paths (empty if no media).
    """
    if not outputs_dir:
        return []

    from cliver.media_handler import MultimediaResponseHandler

    try:
        llm_engine = agent_core.get_llm_engine(model) if agent_core else None
    except Exception:
        llm_engine = None

    handler = MultimediaResponseHandler(save_directory=outputs_dir)
    multimedia = handler.process_response(response, llm_engine=llm_engine)

    if not multimedia.has_media():
        return []

    saved = handler.save_media_content(multimedia, prefix=step_id)
    return saved


async def _validate_step_output(
    subagent,
    step,
    result_text: str,
    media_files: list,
    model: str,
) -> tuple[bool, str]:
    """Use LLM to validate if step output meets expected result.

    Passes generated images to the LLM for visual validation when the model
    supports multimodal input. Audio/video files are described by path.

    Returns (passed, feedback).
    """
    import mimetypes as _mt

    media_section = ""
    image_paths = []
    other_media = []

    if media_files:
        for f in media_files:
            mime = _mt.guess_type(f)[0] or ""
            fname = Path(f).name
            if mime.startswith("image/") and Path(f).exists():
                image_paths.append(f)
                other_media.append(f"- {fname} (image — attached for visual inspection)")
            elif mime.startswith("audio/"):
                other_media.append(f"- {fname} (audio file — verify file exists and format is correct)")
            elif mime.startswith("video/"):
                other_media.append(f"- {fname} (video file — verify file exists and format is correct)")
            else:
                other_media.append(f"- {fname}")
        media_section = f"\n**Media files generated ({len(media_files)}):**\n" + "\n".join(other_media)

    validation_prompt = (
        f"Evaluate if the following step output meets the expected result.\n\n"
        f"**Expected result:** {step.expected_result}\n\n"
        f"**Text output:**\n{result_text[:3000]}\n"
        f"{media_section}\n\n"
        f"Validation rules:\n"
        f"1. Check if the text output satisfies the expected result description\n"
        f"2. If images were expected, verify they were generated (attached for inspection)\n"
        f"3. If audio/video was expected, verify the files exist in the list above\n"
        f"4. Check quantity — if multiple files were expected, verify the count\n\n"
        f"Answer with exactly YES on the first line if ALL expectations are met, "
        f"or NO on the first line followed by specifically what is missing or wrong."
    )

    async def _no_tools(_user_input, _tools):
        return []

    try:
        response = await subagent.process_user_input(
            user_input=validation_prompt,
            model=model,
            images=image_paths if image_paths else None,
            filter_tools=_no_tools,
        )
        from cliver.media_handler import extract_response_text

        answer = extract_response_text(response).strip()
        first_line = answer.split("\n")[0].strip().upper()
        if first_line.startswith("YES"):
            return True, ""
        feedback = answer[answer.find("\n") + 1 :].strip() if "\n" in answer else answer
        return False, feedback
    except Exception as e:
        logger.warning("Validation LLM call failed: %s", e)
        return True, ""


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
            if state.get("error"):
                return {"steps": {step.id: {"outputs": {"error": "Skipped: prior step failed"}, "status": "skipped"}}}

            factory = config["configurable"].get("subagent_factory")
            context = _state_to_execution_context(state)
            rendered_prompt = render_template(step.prompt, context)

            if factory:
                effective_config = agent_config or AgentConfig(model=step.model)
                subagent = factory.create(effective_config)
            else:
                subagent = config["configurable"]["agent_core"]

            def system_appender():
                parts = [_WORKFLOW_STEP_SYSTEM_INSTRUCTION]
                if overview:
                    parts.append(f"# Workflow Overview\n\n{overview}")
                if agent_config:
                    if agent_config.role:
                        parts.append(f"# Your Role\n\n{agent_config.role}")
                    if agent_config.instructions:
                        parts.append(f"# Instructions\n\n{agent_config.instructions}")
                    if agent_config.system_message:
                        parts.append(agent_config.system_message)

                # Inject outputs directory and file context from prior steps
                outputs_dir = state.get("outputs_dir", "")
                file_section = []
                if outputs_dir:
                    file_section.append(f"**Outputs directory:** `{outputs_dir}`")
                    file_section.append("Save ALL generated files to this directory.")
                prior_files = []
                for sid, sdata in state.get("steps", {}).items():
                    s_outputs = sdata.get("outputs", {})
                    if s_outputs.get("media_files"):
                        for f in s_outputs["media_files"]:
                            prior_files.append(f"- `{f}` (from step '{sid}')")
                if prior_files:
                    file_section.append("\n**Files from prior steps:**")
                    file_section.extend(prior_files)
                if file_section:
                    parts.append("# Output Directory & Available Files\n\n" + "\n".join(file_section))

                return "\n\n".join(parts)

            effective_model = (agent_config.model if agent_config else None) or step.model

            # Build tool filter: use agent's explicit tool list if set,
            # otherwise exclude planning/meta tools from workflow steps.
            agent_tools = agent_config.tools if agent_config and agent_config.tools else None

            async def _filter_tools(_user_input, tools):
                if agent_tools:
                    return [t for t in tools if t.name in agent_tools]
                return [t for t in tools if t.name not in _WORKFLOW_EXCLUDED_TOOLS]

            start = time.time()
            deadline = start + step.timeout
            attempt = 0
            current_prompt = rendered_prompt
            result_text = ""
            media_files = []
            validation_status = None

            try:
                from cliver.media_handler import extract_response_text

                while True:
                    response = await subagent.process_user_input(
                        user_input=current_prompt,
                        model=effective_model,
                        system_message_appender=system_appender,
                        filter_tools=_filter_tools,
                        images=render_template(step.images, context) if step.images else None,
                        audio_files=render_template(step.audio_files, context) if step.audio_files else None,
                        video_files=render_template(step.video_files, context) if step.video_files else None,
                        files=render_template(step.files, context) if step.files else None,
                    )

                    result_text = extract_response_text(response)
                    _save_step_output(state["outputs_dir"], step.id, result_text, step.output_format)
                    media_files = _save_step_media(state["outputs_dir"], step.id, response, subagent, effective_model)

                    if not step.expected_result:
                        break

                    passed, feedback = await _validate_step_output(
                        subagent,
                        step,
                        result_text,
                        media_files,
                        effective_model,
                    )
                    if passed:
                        validation_status = "passed"
                        logger.info("Step '%s' validation passed (attempt %d)", step.id, attempt + 1)
                        break

                    attempt += 1
                    validation_status = f"failed (attempt {attempt}): {feedback}"
                    logger.info("Step '%s' validation failed (attempt %d): %s", step.id, attempt, feedback)

                    if step.retry > 0 and attempt >= step.retry:
                        logger.warning("Step '%s' exhausted %d retries", step.id, step.retry)
                        break
                    if time.time() >= deadline:
                        logger.warning("Step '%s' timed out after %ds", step.id, step.timeout)
                        validation_status += " (timeout)"
                        break

                    current_prompt = (
                        rendered_prompt + f"\n\n--- Previous attempt failed validation ---\n"
                        f"Feedback: {feedback}\n"
                        f"Please try again and ensure the output meets: {step.expected_result}"
                    )

                outputs = {"result": result_text, "outputs_dir": state["outputs_dir"]}
                if media_files:
                    outputs["media_files"] = media_files
                if validation_status:
                    outputs["validation"] = validation_status
                if step.outputs:
                    for name in step.outputs:
                        outputs[name] = result_text

                return {
                    "steps": {
                        step.id: {
                            "outputs": outputs,
                            "status": "completed",
                            "execution_time": time.time() - start,
                            "attempts": attempt + 1,
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
            if state.get("error"):
                return {"steps": {step.id: {"outputs": {"error": "Skipped: prior step failed"}, "status": "skipped"}}}
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
            if state.get("error"):
                return {"steps": {step.id: {"outputs": {"error": "Skipped: prior step failed"}, "status": "skipped"}}}
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
            if state.get("error"):
                return {"steps": {step.id: {"outputs": {"error": "Skipped: prior step failed"}, "status": "skipped"}}}
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
