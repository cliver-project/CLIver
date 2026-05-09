# tests/test_workflow_compiler.py
from cliver.workflow.compiler import WorkflowCompiler, merge_steps
from cliver.workflow.workflow_models import Workflow


class TestMergeSteps:
    def test_merge_none_left(self):
        assert merge_steps(None, {"s1": {"r": 1}}) == {"s1": {"r": 1}}

    def test_merge_accumulates(self):
        left = {"s1": {"r": 1}}
        right = {"s2": {"r": 2}}
        merged = merge_steps(left, right)
        assert merged == {"s1": {"r": 1}, "s2": {"r": 2}}

    def test_merge_overwrites(self):
        left = {"s1": {"r": 1}}
        right = {"s1": {"r": 2}}
        merged = merge_steps(left, right)
        assert merged == {"s1": {"r": 2}}


class TestWorkflowCompiler:
    def test_compile_linear_workflow(self):
        wf = Workflow(
            name="linear",
            steps=[
                {"id": "s1", "type": "llm", "prompt": "Step 1"},
                {"id": "s2", "type": "llm", "prompt": "Step 2", "depends_on": ["s1"]},
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None

    def test_compile_parallel_workflow(self):
        wf = Workflow(
            name="parallel",
            steps=[
                {"id": "s1", "type": "llm", "prompt": "A"},
                {"id": "s2", "type": "llm", "prompt": "B"},
                {"id": "s3", "type": "llm", "prompt": "C", "depends_on": ["s1", "s2"]},
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None

    def test_compile_with_conditions(self):
        wf = Workflow(
            name="branching",
            steps=[
                {"id": "classify", "type": "llm", "prompt": "Classify"},
                {
                    "id": "positive",
                    "type": "llm",
                    "prompt": "Good",
                    "depends_on": ["classify"],
                    "condition": "classify.sentiment == 'positive'",
                },
                {
                    "id": "negative",
                    "type": "llm",
                    "prompt": "Bad",
                    "depends_on": ["classify"],
                    "condition": "classify.sentiment == 'negative'",
                },
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None

    def test_compile_mixed_types(self):
        wf = Workflow(
            name="mixed",
            steps=[
                {"id": "s1", "type": "llm", "prompt": "Start"},
                {"id": "s2", "type": "python", "file": "./run.py", "depends_on": ["s1"]},
            ],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None

    def test_compile_single_step(self):
        wf = Workflow(
            name="single",
            steps=[{"id": "only", "type": "llm", "prompt": "Do it"}],
        )
        compiler = WorkflowCompiler()
        graph = compiler.compile(wf)
        assert graph is not None
