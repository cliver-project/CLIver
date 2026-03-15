"""
Base classes for workflow step executors.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from cliver import template_utils
from cliver.workflow.workflow_models import BaseStep, ExecutionContext, ExecutionResult

logger = logging.getLogger(__name__)


class StepExecutor(ABC):
    """Abstract base class for step executors."""

    def __init__(self, step: BaseStep):
        self.step = step
        self._jinja_env = template_utils.get_jinja_env()

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute the step with the given context."""
        pass

    def resolve_variable(self, value: Any, context: ExecutionContext) -> Any:
        """Resolve variable references using Jinja2 templating.

        Context hierarchy (Jinja2 handles nested access natively):
        - inputs.* — workflow inputs
        - {step_id}.outputs.* — previous step outputs
        - Environment variables as fallback
        """
        if isinstance(value, str):
            if "{{" not in value:
                return value

            # Build context dict — Jinja2 handles nested dict access natively
            template_context = {}

            # Workflow inputs — available as both inputs.key and key (shorthand)
            if context.inputs:
                template_context["inputs"] = context.inputs
                template_context.update(context.inputs)

            # Step outputs: each step_id maps to its result dict
            for step_id, step_data in context.steps.items():
                template_context[step_id] = step_data

            # Environment variables as fallback
            template_context.update(os.environ)

            try:
                template = self._jinja_env.from_string(value)
                return template.render(**template_context)
            except Exception as e:
                logger.warning(f"Error resolving template '{value}': {e}")
                return value
        elif isinstance(value, dict):
            return {k: self.resolve_variable(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve_variable(v, context) for v in value]
        else:
            return value

    async def extract_outputs(self, result) -> dict[Any, Any]:
        """Extract named outputs from a step result."""
        outputs = {}
        if self.step.outputs:
            if len(self.step.outputs) == 1:
                outputs[self.step.outputs[0]] = result
            elif isinstance(result, dict):
                for name in self.step.outputs:
                    if name in result:
                        outputs[name] = result[name]
            elif isinstance(result, (list, tuple)):
                for i, name in enumerate(self.step.outputs):
                    if i < len(result):
                        outputs[name] = result[i]
            else:
                for name in self.step.outputs:
                    outputs[name] = result
        return outputs
