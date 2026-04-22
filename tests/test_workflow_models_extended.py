"""Tests for extended workflow models: AgentConfig, overview, output_format."""

import yaml

from cliver.workflow.workflow_models import (
    AgentConfig,
    LLMStep,
    Workflow,
)


class TestAgentConfig:
    def test_all_fields_optional(self):
        cfg = AgentConfig()
        assert cfg.model is None
        assert cfg.system_message is None
        assert cfg.tools is None
        assert cfg.skills is None
        assert cfg.permissions is None

    def test_full_config(self):
        cfg = AgentConfig(
            model="deepseek-r1",
            system_message="You are an architect.",
            tools=["read_file", "grep_search"],
            skills=["brainstorm"],
            permissions={"mode": "auto-edit"},
        )
        assert cfg.model == "deepseek-r1"
        assert cfg.tools == ["read_file", "grep_search"]


class TestWorkflowOverview:
    def test_inline_overview(self):
        wf = Workflow(name="test", overview="## Project\nA test project.", steps=[])
        assert "test project" in wf.overview

    def test_overview_file(self):
        wf = Workflow(name="test", overview_file="path/to/overview.md", steps=[])
        assert wf.overview_file == "path/to/overview.md"

    def test_no_overview(self):
        wf = Workflow(name="test", steps=[])
        assert wf.overview is None
        assert wf.overview_file is None


class TestWorkflowAgents:
    def test_agents_section(self):
        wf = Workflow(
            name="test",
            agents={"architect": AgentConfig(model="deepseek-r1"), "developer": AgentConfig(model="qwen")},
            steps=[],
        )
        assert "architect" in wf.agents
        assert wf.agents["architect"].model == "deepseek-r1"


class TestLLMStepExtended:
    def test_agent_reference(self):
        step = LLMStep(id="design", name="Design", prompt="Design it", agent="architect")
        assert step.agent == "architect"

    def test_output_format_default(self):
        step = LLMStep(id="s1", name="S1", prompt="Do it")
        assert step.output_format == "md"

    def test_output_format_custom(self):
        step = LLMStep(id="s1", name="S1", prompt="Do it", output_format="json")
        assert step.output_format == "json"


class TestWorkflowOutputsDir:
    def test_outputs_dir(self):
        wf = Workflow(name="test", outputs_dir=".cliver/workflow-runs/test", steps=[])
        assert wf.outputs_dir == ".cliver/workflow-runs/test"

    def test_outputs_dir_default_none(self):
        wf = Workflow(name="test", steps=[])
        assert wf.outputs_dir is None


class TestYAMLRoundtrip:
    def test_workflow_with_agents_from_yaml(self):
        yaml_str = """
name: pipeline
overview: "A test pipeline"
agents:
  dev:
    model: qwen
    tools: [read_file, write_file]
steps:
  - id: s1
    name: Step 1
    type: llm
    prompt: "Do something"
    agent: dev
    output_format: json
"""
        data = yaml.safe_load(yaml_str)
        wf = Workflow(**data)
        assert wf.agents["dev"].model == "qwen"
        assert wf.steps[0].agent == "dev"
        assert wf.steps[0].output_format == "json"
