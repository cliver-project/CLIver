"""Tests for workflow model definitions."""

import pytest
from pydantic import ValidationError

from cliver.workflow.workflow_models import (
    Branch,
    DecisionStep,
    ExecutionContext,
    ExecutionResult,
    FunctionStep,
    HumanStep,
    LLMStep,
    StepType,
    Workflow,
    WorkflowExecutionState,
    WorkflowStep,
)


class TestStepTypes:
    def test_llm_step(self):
        step = LLMStep(id="s1", name="Ask LLM", prompt="Hello")
        assert step.type == StepType.LLM
        assert step.prompt == "Hello"
        assert step.depends_on == []
        assert step.condition is None
        assert step.retry == 0

    def test_function_step(self):
        step = FunctionStep(id="s2", name="Run func", function="mymodule.func")
        assert step.type == StepType.FUNCTION

    def test_human_step(self):
        step = HumanStep(id="s3", name="Ask user", prompt="Confirm?")
        assert step.type == StepType.HUMAN

    def test_workflow_step(self):
        step = WorkflowStep(id="s4", name="Sub workflow", workflow="deploy")
        assert step.type == StepType.WORKFLOW

    def test_decision_step(self):
        step = DecisionStep(
            id="s5",
            name="Branch",
            branches=[
                Branch(condition="True", next_step="s1"),
                Branch(condition="False", next_step="s2"),
            ],
            default="s2",
        )
        assert step.type == StepType.DECISION
        assert len(step.branches) == 2
        assert step.default == "s2"

    def test_step_with_depends_on(self):
        step = LLMStep(id="s1", name="Step", prompt="p", depends_on=["s0"])
        assert step.depends_on == ["s0"]

    def test_step_with_condition(self):
        step = LLMStep(id="s1", name="Step", prompt="p", condition="True")
        assert step.condition == "True"

    def test_step_with_retry(self):
        step = LLMStep(id="s1", name="Step", prompt="p", retry=3)
        assert step.retry == 3


class TestWorkflow:
    def test_create_workflow(self):
        wf = Workflow(
            name="test",
            description="Test workflow",
            steps=[LLMStep(id="s1", name="Step 1", prompt="Hello")],
        )
        assert wf.name == "test"
        assert len(wf.steps) == 1

    def test_workflow_with_inputs(self):
        wf = Workflow(
            name="test",
            inputs={"branch": "main"},
            steps=[],
        )
        result = wf.get_initial_inputs({"branch": "dev"})
        assert result["branch"] == "dev"

    def test_workflow_default_inputs(self):
        wf = Workflow(
            name="test",
            inputs={"branch": "main", "env": "staging"},
            steps=[],
        )
        result = wf.get_initial_inputs(None)
        assert result["branch"] == "main"
        assert result["env"] == "staging"


class TestExecutionContext:
    def test_empty_context(self):
        ctx = ExecutionContext(workflow_name="test")
        assert ctx.inputs == {}
        assert ctx.steps == {}

    def test_context_with_step_outputs(self):
        ctx = ExecutionContext(
            workflow_name="test",
            steps={"s1": {"outputs": {"result": "hello"}, "status": "completed"}},
        )
        assert ctx.steps["s1"]["outputs"]["result"] == "hello"


class TestExecutionResult:
    def test_success_result(self):
        r = ExecutionResult(step_id="s1", outputs={"result": "ok"})
        assert r.success is True

    def test_failure_result(self):
        r = ExecutionResult(step_id="s1", success=False, error="boom")
        assert r.error == "boom"


class TestWorkflowExecutionState:
    def test_initial_state(self):
        state = WorkflowExecutionState(
            workflow_name="test",
            execution_id="abc123",
            context=ExecutionContext(workflow_name="test"),
        )
        assert state.status == "running"
        assert state.completed_steps == []
        assert state.skipped_steps == []

    def test_state_statuses(self):
        state = WorkflowExecutionState(
            workflow_name="test",
            execution_id="abc123",
            context=ExecutionContext(workflow_name="test"),
            status="paused",
        )
        assert state.status == "paused"


class TestWorkflowStepFile:
    def test_workflow_step_with_name(self):
        step = WorkflowStep(id="s1", name="Sub", workflow="deploy")
        assert step.workflow == "deploy"
        assert step.workflow_file is None

    def test_workflow_step_with_file(self):
        step = WorkflowStep(id="s1", name="Sub", workflow_file="sub-workflows/deploy.yaml")
        assert step.workflow_file == "sub-workflows/deploy.yaml"
        assert step.workflow is None

    def test_workflow_step_both_allowed(self):
        step = WorkflowStep(id="s1", name="Sub", workflow="deploy", workflow_file="deploy.yaml")
        assert step.workflow == "deploy"
        assert step.workflow_file == "deploy.yaml"

    def test_workflow_step_neither_raises(self):
        with pytest.raises(ValidationError, match="At least one"):
            WorkflowStep(id="s1", name="Sub")

    def test_workflow_step_with_inputs(self):
        step = WorkflowStep(
            id="s1",
            name="Sub",
            workflow_file="sub.yaml",
            workflow_inputs={"key": "{{ inputs.value }}"},
        )
        assert step.workflow_inputs == {"key": "{{ inputs.value }}"}


class TestWorkflowCreateSkill:
    def test_skill_exists(self):
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        names = manager.get_skill_names()
        assert "workflow-create" in names

    def test_skill_has_content(self):
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        skill = manager.get_skill("workflow-create")
        assert skill is not None
        assert skill.body and len(skill.body) > 100


class TestWBSPlannerSkill:
    def test_skill_exists(self):
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        names = manager.get_skill_names()
        assert "wbs-planner" in names

    def test_skill_has_content(self):
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        skill = manager.get_skill("wbs-planner")
        assert skill is not None
        assert skill.body and len(skill.body) > 100

    def test_skill_allowed_tools(self):
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        skill = manager.get_skill("wbs-planner")
        assert skill is not None
        assert "WorkflowValidate" in skill.allowed_tools
        assert "Write" in skill.allowed_tools
        assert "Ask" in skill.allowed_tools
