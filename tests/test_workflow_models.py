# tests/test_workflow_models.py
import pytest

from cliver.workflow.workflow_models import (
    LLMStep,
    PythonStep,
    Workflow,
)


class TestLLMStep:
    def test_minimal_llm_step(self):
        step = LLMStep(id="s1", prompt="Hello")
        assert step.type == "llm"
        assert step.prompt == "Hello"
        assert step.model is None
        assert step.role is None
        assert step.tools is None
        assert step.output_format == "json"
        assert step.depends_on == []
        assert step.condition is None

    def test_full_llm_step(self):
        step = LLMStep(
            id="research",
            prompt="Research ${inputs.topic}",
            model="qwen",
            role="Research analyst",
            tools=["web_search", "read_file"],
            output_format="markdown",
            depends_on=["prior"],
            condition="prior.done == true",
        )
        assert step.model == "qwen"
        assert step.role == "Research analyst"
        assert step.tools == ["web_search", "read_file"]
        assert step.output_format == "markdown"
        assert step.depends_on == ["prior"]
        assert step.condition == "prior.done == true"

    def test_llm_step_invalid_output_format(self):
        with pytest.raises(ValueError):
            LLMStep(id="s1", prompt="Hi", output_format="xml")


class TestPythonStep:
    def test_minimal_python_step(self):
        step = PythonStep(id="s1", file="./scripts/run.py")
        assert step.type == "python"
        assert step.file == "./scripts/run.py"
        assert step.depends_on == []

    def test_python_step_with_deps(self):
        step = PythonStep(
            id="transform",
            file="./transform.py",
            depends_on=["research"],
            condition="research.count > 0",
        )
        assert step.depends_on == ["research"]
        assert step.condition == "research.count > 0"


class TestWorkflow:
    def test_minimal_workflow(self):
        wf = Workflow(
            name="test-wf",
            steps=[{"id": "s1", "type": "llm", "prompt": "Hello"}],
        )
        assert wf.name == "test-wf"
        assert len(wf.steps) == 1
        assert isinstance(wf.steps[0], LLMStep)
        assert wf.inputs is None
        assert wf.permissions is None

    def test_workflow_with_inputs(self):
        wf = Workflow(
            name="test",
            inputs={
                "topic": {"type": "string", "description": "Topic"},
                "lang": {"type": "string", "default": "English"},
            },
            steps=[{"id": "s1", "type": "llm", "prompt": "Hello"}],
        )
        assert wf.inputs["lang"]["default"] == "English"

    def test_workflow_mixed_step_types(self):
        wf = Workflow(
            name="mixed",
            steps=[
                {"id": "s1", "type": "llm", "prompt": "Analyze"},
                {"id": "s2", "type": "python", "file": "./run.py", "depends_on": ["s1"]},
            ],
        )
        assert isinstance(wf.steps[0], LLMStep)
        assert isinstance(wf.steps[1], PythonStep)

    def test_workflow_rejects_unknown_step_type(self):
        with pytest.raises(ValueError):
            Workflow(
                name="bad",
                steps=[{"id": "s1", "type": "human", "prompt": "Hello"}],
            )

    def test_get_initial_inputs_defaults(self):
        wf = Workflow(
            name="test",
            inputs={
                "topic": {"type": "string", "default": "AI"},
                "lang": {"type": "string"},
            },
            steps=[{"id": "s1", "type": "llm", "prompt": "Hi"}],
        )
        result = wf.get_initial_inputs({"lang": "French"})
        assert result == {"topic": "AI", "lang": "French"}

    def test_get_initial_inputs_override(self):
        wf = Workflow(
            name="test",
            inputs={"topic": {"type": "string", "default": "AI"}},
            steps=[{"id": "s1", "type": "llm", "prompt": "Hi"}],
        )
        result = wf.get_initial_inputs({"topic": "ML"})
        assert result == {"topic": "ML"}

    def test_workflow_from_yaml(self):
        import yaml

        yaml_str = """
name: test-wf
description: A test workflow
inputs:
  topic:
    type: string
    default: AI
steps:
  - id: research
    type: llm
    model: qwen
    role: "Analyst"
    prompt: "Research ${inputs.topic}"
    output_format: json
  - id: transform
    type: python
    file: ./transform.py
    depends_on: [research]
"""
        data = yaml.safe_load(yaml_str)
        wf = Workflow(**data)
        assert wf.name == "test-wf"
        assert isinstance(wf.steps[0], LLMStep)
        assert isinstance(wf.steps[1], PythonStep)
        assert wf.steps[0].model == "qwen"
        assert wf.steps[1].depends_on == ["research"]
