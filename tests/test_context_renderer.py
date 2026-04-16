"""Tests for Jinja2 context rendering in workflow steps."""

import pytest

from cliver.workflow.context_renderer import evaluate_condition, render_template
from cliver.workflow.workflow_models import ExecutionContext


@pytest.fixture
def context_with_outputs():
    return ExecutionContext(
        workflow_name="test",
        inputs={"branch": "main", "env": "staging"},
        steps={
            "run_tests": {
                "outputs": {"test_result": "PASSED: 42/42", "coverage": "85%"},
                "status": "completed",
            },
            "build": {
                "outputs": {"artifact": "app.tar.gz"},
                "status": "completed",
            },
        },
    )


class TestRenderTemplate:
    def test_simple_input_reference(self, context_with_outputs):
        result = render_template("Deploy to {{ inputs.branch }}", context_with_outputs)
        assert result == "Deploy to main"

    def test_step_output_reference(self, context_with_outputs):
        result = render_template(
            "Results: {{ run_tests.outputs.test_result }}",
            context_with_outputs,
        )
        assert result == "Results: PASSED: 42/42"

    def test_multiple_references(self, context_with_outputs):
        result = render_template(
            "{{ inputs.branch }} - {{ build.outputs.artifact }}",
            context_with_outputs,
        )
        assert result == "main - app.tar.gz"

    def test_no_template_passthrough(self, context_with_outputs):
        result = render_template("plain text", context_with_outputs)
        assert result == "plain text"

    def test_missing_variable_returns_empty(self, context_with_outputs):
        result = render_template("{{ nonexistent.outputs.foo }}", context_with_outputs)
        assert result == ""

    def test_render_dict_values(self, context_with_outputs):
        d = {"key": "{{ inputs.branch }}", "static": "hello"}
        result = render_template(d, context_with_outputs)
        assert result == {"key": "main", "static": "hello"}

    def test_render_list_values(self, context_with_outputs):
        lst = ["{{ inputs.branch }}", "static"]
        result = render_template(lst, context_with_outputs)
        assert result == ["main", "static"]


class TestEvaluateCondition:
    def test_true_condition(self, context_with_outputs):
        assert evaluate_condition("'PASSED' in run_tests.outputs.test_result", context_with_outputs) is True

    def test_false_condition(self, context_with_outputs):
        assert evaluate_condition("'FAILED' in run_tests.outputs.test_result", context_with_outputs) is False

    def test_input_condition(self, context_with_outputs):
        assert evaluate_condition("inputs.env == 'staging'", context_with_outputs) is True

    def test_none_condition_is_true(self, context_with_outputs):
        assert evaluate_condition(None, context_with_outputs) is True

    def test_empty_condition_is_true(self, context_with_outputs):
        assert evaluate_condition("", context_with_outputs) is True

    def test_error_condition_is_false(self, context_with_outputs):
        assert evaluate_condition("undefined_var.foo", context_with_outputs) is False
