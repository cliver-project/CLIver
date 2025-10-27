"""
Base classes for workflow steps.
"""

import asyncio
import logging
import os
import time
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
        """Execute the step with the given context.

        Args:
            context: Execution context containing inputs and outputs

        Returns:
            ExecutionResult with outputs and success status
        """
        pass

    async def execute_with_retry(self, context: ExecutionContext) -> ExecutionResult:
        """Execute the step with retry logic if configured."""
        if not self.step.retry:
            return await self.execute(context)

        max_attempts = self.step.retry.max_attempts
        backoff_factor = self.step.retry.backoff_factor
        max_backoff = self.step.retry.max_backoff

        last_exception = None
        for attempt in range(max_attempts):
            try:
                start_time = time.time()
                result = await self.execute(context)
                result.execution_time = time.time() - start_time
                return result
            except Exception as e:
                last_exception = e
                logger.warning(f"Step {self.step.id} attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_attempts - 1:  # Not the last attempt
                    # Calculate backoff with exponential factor
                    backoff_time = min(backoff_factor * (2**attempt), max_backoff)
                    logger.info(f"Retrying step {self.step.id} in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    # an exception is thrown after retries
                    raise Exception(f"Step {self.step.id} failed after {max_attempts} attempts") from last_exception

        # If we get here, all retries failed
        return ExecutionResult(
            step_id=self.step.id,
            success=False,
            error=f"Step failed after {max_attempts} attempts: {str(last_exception)}",
            execution_time=0.0,
        )

    def evaluate_condition(self, context: ExecutionContext) -> bool:
        """Evaluate the step condition if present.

        Args:
            context: Execution context

        Returns:
            True if condition is met or no condition, False otherwise
        """
        if not self.step.condition:
            return True

        # Use Jinja2 template resolution for condition evaluation
        # resolved_condition = self.resolve_variable(self.step.condition, context)

        # We don't know how to evaluate yet.
        return True

    def resolve_variable(self, value: Any, context: ExecutionContext) -> Any:
        """Resolve variable references in input values using Jinja2 templating.

        Args:
            value: Input value that may contain variable references
            context: Execution context

        Returns:
            Resolved value
        """
        if isinstance(value, str):
            # Create a context dict for variable resolution with new hierarchical structure
            template_context = {}

            # Add workflow inputs at the root level
            if context.inputs:
                template_context["inputs"] = context.inputs

            # Build the new hierarchical structure for step access from the steps dict
            step_entries = {}
            for step_id, step_info in context.steps.items():
                if step_id not in step_entries:
                    step_entries[step_id] = {}
                # Add step inputs if available
                if step_info.inputs:
                    step_entries[step_id]["inputs"] = step_info.inputs
                # Add step outputs if available
                if step_info.outputs:
                    step_entries[step_id]["outputs"] = step_info.outputs

            # Merge step entries into template context
            template_context.update(step_entries)

            # Add environment variables as fallback
            template_context.update(os.environ)

            # Handle nested dictionary access with dot notation
            extended_context = template_context.copy()

            def _add_nested_keys(prefix, obj):
                """Recursively add nested keys with dot notation."""
                if isinstance(obj, dict):
                    # Add the dictionary itself as a key (for accessing the whole dict)
                    if prefix:
                        extended_context[prefix] = obj

                    # Add nested keys with dot notation
                    for nested_key, nested_val in obj.items():
                        new_key = f"{prefix}.{nested_key}" if prefix else nested_key
                        extended_context[new_key] = nested_val
                        # If the nested value is also a dict, recursively process it
                        if isinstance(nested_val, dict):
                            _add_nested_keys(new_key, nested_val)

            # Add all nested keys for variables, outputs, and inputs
            for key, val in template_context.items():
                if isinstance(val, dict):
                    _add_nested_keys(key, val)

            # Also process the root level for direct access to inputs
            if "inputs" in template_context and isinstance(template_context["inputs"], dict):
                for key, val in template_context["inputs"].items():
                    extended_context[key] = val
                    # Process nested keys for inputs as well
                    if isinstance(val, dict):
                        _add_nested_keys(key, val)

            # Use Jinja2 templating for variable resolution
            try:
                # Use the reusable Jinja2 environment
                template = self._jinja_env.from_string(value)
                return template.render(**extended_context)
            except Exception as e:
                logger.warning(f"Error resolving Jinja2 template '{value}': {str(e)}")
                return value
        elif isinstance(value, dict):
            # Recursively resolve dict values
            return {k: self.resolve_variable(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively resolve list values
            return [self.resolve_variable(v, context) for v in value]
        else:
            return value

    async def extract_outputs(self, result) -> dict[Any, Any]:
        outputs = {}
        if self.step.outputs:
            if len(self.step.outputs) == 1:
                outputs[self.step.outputs[0]] = result
            else:
                if isinstance(result, dict):
                    # If result is dict and multiple outputs, extract specified keys that only
                    # exists in the step.outputs definitions
                    for output_name in self.step.outputs:
                        if output_name in result:
                            outputs[output_name] = result[output_name]
                elif isinstance(result, (list, tuple)):
                    for i, output_name in enumerate(self.step.outputs):
                        if i < len(result):
                            outputs[output_name] = result[i]
                else:
                    # not a dict/list/tuple result, and we expect it should be because we
                    # have multiple expected outputs
                    #    we just assign the result to all the expected outputs.
                    # Multiple outputs with non-dict result - this is ambiguous
                    logger.warning(
                        f"Ambiguous output mapping for step {self.step.id}. "
                        f"Function returned {type(result)}, but {len(self.step.outputs)} outputs specified."
                    )
                    for output_name in self.step.outputs:
                        outputs[output_name] = result
        return outputs
