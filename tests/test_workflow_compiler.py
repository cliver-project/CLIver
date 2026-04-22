"""Tests for WorkflowCompiler — YAML Workflow -> LangGraph StateGraph."""

from cliver.workflow.compiler import WorkflowCompiler, merge_steps
from cliver.workflow.workflow_models import (
    AgentConfig,
    Branch,
    DecisionStep,
    FunctionStep,
    HumanStep,
    LLMStep,
    Workflow,
)


class TestMergeSteps:
    def test_merge_new_keys(self):
        existing = {"step1": {"outputs": {"result": "a"}, "status": "completed"}}
        new = {"step2": {"outputs": {"result": "b"}, "status": "completed"}}
        merged = merge_steps(existing, new)
        assert "step1" in merged
        assert "step2" in merged

    def test_merge_preserves_existing(self):
        existing = {"step1": {"outputs": {"result": "a"}}}
        new = {"step1": {"outputs": {"result": "b"}}}
        merged = merge_steps(existing, new)
        assert merged["step1"]["outputs"]["result"] == "b"

    def test_merge_empty(self):
        assert merge_steps({}, {}) == {}

    def test_merge_none_left(self):
        result = merge_steps(None, {"a": 1})
        assert result == {"a": 1}


class TestCompileLinear:
    def test_compile_linear_workflow(self):
        wf = Workflow(
            name="test",
            steps=[
                LLMStep(id="s1", name="Step 1", prompt="Do A"),
                LLMStep(id="s2", name="Step 2", prompt="Do B", depends_on=["s1"]),
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None

    def test_compile_with_agents(self):
        wf = Workflow(
            name="test",
            agents={"dev": AgentConfig(model="qwen")},
            steps=[
                LLMStep(id="s1", name="S1", prompt="Do it", agent="dev"),
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None


class TestCompileParallel:
    def test_compile_fan_out_fan_in(self):
        wf = Workflow(
            name="test",
            steps=[
                LLMStep(id="start", name="Start", prompt="Begin"),
                LLMStep(id="branch_a", name="A", prompt="Do A", depends_on=["start"]),
                LLMStep(id="branch_b", name="B", prompt="Do B", depends_on=["start"]),
                LLMStep(id="merge", name="Merge", prompt="Combine", depends_on=["branch_a", "branch_b"]),
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None


class TestCompileDecision:
    def test_compile_decision_step(self):
        wf = Workflow(
            name="test",
            steps=[
                LLMStep(id="analyze", name="Analyze", prompt="Analyze"),
                DecisionStep(
                    id="decide",
                    name="Decide",
                    branches=[
                        Branch(condition="analyze.outputs.result == 'yes'", next_step="path_a"),
                        Branch(condition="True", next_step="path_b"),
                    ],
                    default="path_b",
                    depends_on=["analyze"],
                ),
                LLMStep(id="path_a", name="Path A", prompt="A", depends_on=["decide"]),
                LLMStep(id="path_b", name="Path B", prompt="B", depends_on=["decide"]),
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None


class TestCompileHuman:
    def test_compile_human_step(self):
        wf = Workflow(
            name="test",
            steps=[
                HumanStep(id="review", name="Review", prompt="Approve?"),
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None


class TestCompileFunction:
    def test_compile_function_step(self):
        wf = Workflow(
            name="test",
            steps=[
                FunctionStep(id="f1", name="Func", function="os.getcwd"),
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None
