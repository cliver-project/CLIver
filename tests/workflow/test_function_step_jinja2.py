"""
Tests for Jinja2 template support in function step.
"""
import os
from cliver.workflow.steps.function_step import FunctionStepExecutor
from cliver.workflow.workflow_models import FunctionStep, ExecutionContext


def test_jinja2_template_resolution():
    """Test that Jinja2 templates are resolved correctly."""
    # Create a mock step
    step = FunctionStep(
        id="test_step",
        name="Test Step",
        function="test_module.test_function",
        inputs={"param": "Hello {{ name }}!"},
        outputs=["result"]
    )

    # Create executor
    executor = FunctionStepExecutor(step)

    # Create context with variables
    context = ExecutionContext(
        workflow_name="test_workflow",
        inputs={"name": "World"}
    )

    # Test template resolution
    resolved = executor.resolve_variable("Hello {{ name }}!", context)
    assert resolved == "Hello World!"


def test_jinja2_nested_context_resolution():
    """Test that nested context values are resolved with dot notation."""
    # Create a mock step
    step = FunctionStep(
        id="test_step",
        name="Test Step",
        function="test_module.test_function",
        inputs={"param": "Value: {{ data.key }}"},
        outputs=["result"]
    )

    # Create executor
    executor = FunctionStepExecutor(step)

    # Create context with nested outputs
    context = ExecutionContext(
        workflow_name="test_workflow",
        inputs={"data": {"key": "nested_value"}}
    )

    # Test nested template resolution
    resolved = executor.resolve_variable("Value: {{ data.key }}", context)
    assert resolved == "Value: nested_value"


def test_jinja2_environment_variable_fallback():
    """Test that environment variables are used as fallback."""
    # Create a mock step
    step = FunctionStep(
        id="test_step",
        name="Test Step",
        function="test_module.test_function",
        inputs={"param": "Env: {{ TEST_ENV_VAR }}"},
        outputs=["result"]
    )

    # Create executor
    executor = FunctionStepExecutor(step)

    # Set environment variable
    os.environ["TEST_ENV_VAR"] = "env_value"

    # Create context without the variable
    context = ExecutionContext(
        workflow_name="test_workflow",
        inputs={}
    )

    # Test environment variable fallback
    resolved = executor.resolve_variable("Env: {{ TEST_ENV_VAR }}", context)
    assert resolved == "Env: env_value"

    # Clean up
    del os.environ["TEST_ENV_VAR"]


def test_jinja2_deeply_nested_context():
    """Test deeply nested context resolution."""
    # Create a mock step
    step = FunctionStep(
        id="test_step",
        name="Test Step",
        function="test_module.test_function",
        inputs={"param": "{{ a.b.c.d }}"},
        outputs=["result"]
    )

    # Create executor
    executor = FunctionStepExecutor(step)

    # Create context with deeply nested outputs
    context = ExecutionContext(
        workflow_name="test_workflow",
        inputs={"a": {"b": {"c": {"d": "deep_value"}}}}
    )

    # Test deeply nested template resolution
    resolved = executor.resolve_variable("{{ a.b.c.d }}", context)
    assert resolved == "deep_value"