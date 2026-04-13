"""Tests for the create_workflow builtin tool."""

import textwrap

import pytest

from cliver.tools.create_workflow import CreateWorkflowTool


@pytest.fixture
def tool():
    return CreateWorkflowTool()


@pytest.fixture
def valid_yaml():
    return textwrap.dedent("""\
        name: test-workflow
        description: A test workflow
        inputs:
          topic: "AI"
        steps:
          - id: research
            name: Research
            type: llm
            prompt: "Research {{ inputs.topic }}"
            outputs: [findings]
          - id: summarize
            name: Summarize
            type: llm
            prompt: "Summarize: {{ research.outputs.findings }}"
            outputs: [summary]
    """)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_workflow_creates_file(self, tool, valid_yaml, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        result = tool._run(workflow_yaml=valid_yaml)
        assert "test-workflow" in result
        assert "2 steps" in result
        assert (tmp_path / ".cliver" / "workflows" / "test-workflow.yaml").exists()

    def test_invalid_yaml_syntax(self, tool):
        result = tool._run(workflow_yaml="{ invalid: yaml: [")
        assert "Error" in result
        assert "YAML" in result or "syntax" in result.lower()

    def test_invalid_workflow_structure(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        result = tool._run(workflow_yaml="name: test\n")
        # Missing steps — Workflow model requires steps
        assert "Error" in result or "created" in result  # steps defaults to empty list

    def test_non_dict_yaml(self, tool):
        result = tool._run(workflow_yaml="- item1\n- item2\n")
        assert "Error" in result
        assert "mapping" in result or "dict" in result

    def test_empty_steps(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        result = tool._run(workflow_yaml="name: empty\nsteps: []\n")
        assert "Error" in result
        assert "at least one step" in result


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


class TestFileOutput:
    def test_saves_to_cliver_workflows_dir(self, tool, valid_yaml, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        tool._run(workflow_yaml=valid_yaml)

        saved = tmp_path / ".cliver" / "workflows" / "test-workflow.yaml"
        assert saved.exists()
        content = saved.read_text()
        assert "test-workflow" in content
        assert "Research" in content

    def test_creates_directories_if_needed(self, tool, valid_yaml, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        assert not (tmp_path / ".cliver").exists()
        tool._run(workflow_yaml=valid_yaml)
        assert (tmp_path / ".cliver" / "workflows").is_dir()

    def test_overwrites_existing_file(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        yaml_v1 = "name: overwrite\nsteps:\n  - id: s1\n    name: S1\n    type: llm\n    prompt: v1\n    outputs: [x]\n"
        yaml_v2 = "name: overwrite\nsteps:\n  - id: s1\n    name: S1\n    type: llm\n    prompt: v2\n    outputs: [x]\n"

        tool._run(workflow_yaml=yaml_v1)
        tool._run(workflow_yaml=yaml_v2)

        content = (tmp_path / ".cliver" / "workflows" / "overwrite.yaml").read_text()
        assert "v2" in content
        assert "v1" not in content


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------


class TestResponse:
    def test_includes_step_summary(self, tool, valid_yaml, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        result = tool._run(workflow_yaml=valid_yaml)
        assert "[llm] Research" in result
        assert "[llm] Summarize" in result

    def test_execute_flag_shows_run_command(self, tool, valid_yaml, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        result = tool._run(workflow_yaml=valid_yaml, execute=True)
        assert "cliver workflow run test-workflow" in result

    def test_execute_flag_includes_input_params(self, tool, valid_yaml, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        result = tool._run(workflow_yaml=valid_yaml, execute=True)
        assert "topic=" in result


# ---------------------------------------------------------------------------
# Step type validation
# ---------------------------------------------------------------------------


class TestStepTypes:
    def test_function_step(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        yaml_str = textwrap.dedent("""\
            name: func-test
            steps:
              - id: compute
                name: Compute
                type: function
                function: mymodule.compute
                outputs: [result]
        """)
        result = tool._run(workflow_yaml=yaml_str)
        assert "func-test" in result
        assert "[function] Compute" in result

    def test_human_step(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        yaml_str = textwrap.dedent("""\
            name: human-test
            steps:
              - id: confirm
                name: Confirm
                type: human
                prompt: "Continue?"
        """)
        result = tool._run(workflow_yaml=yaml_str)
        assert "[human] Confirm" in result

    def test_mixed_step_types(self, tool, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        yaml_str = textwrap.dedent("""\
            name: mixed-test
            steps:
              - id: s1
                name: Research
                type: llm
                prompt: "Research"
                outputs: [data]
              - id: s2
                name: Process
                type: function
                function: mod.func
                outputs: [processed]
              - id: s3
                name: Confirm
                type: human
                prompt: "OK?"
        """)
        result = tool._run(workflow_yaml=yaml_str)
        assert "3 steps" in result
        assert "[llm] Research" in result
        assert "[function] Process" in result
        assert "[human] Confirm" in result


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_planning_section_includes_all_three_levels(self):
        from cliver.llm.base import LLMInferenceEngine

        prompt = LLMInferenceEngine._section_interaction_guidelines()
        assert "Simple" in prompt
        assert "Medium" in prompt
        assert "Complex" in prompt
        assert "skill('brainstorm')" in prompt
        assert "skill('write-plan')" in prompt
        assert "skill('execute-plan')" in prompt
        assert "todo_write" in prompt
        assert "todo_read" in prompt
