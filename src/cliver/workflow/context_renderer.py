"""
Jinja2-based context renderer for workflow step templates.

Renders step prompts, inputs, and conditions against the ExecutionContext,
enabling step-to-step result propagation via {{ step_id.outputs.key }} syntax.
"""

import logging
from typing import Any

from jinja2 import Environment, Undefined

from cliver.workflow.workflow_models import ExecutionContext

logger = logging.getLogger(__name__)


class _PermissiveUndefined(Undefined):
    """Undefined subclass that silently returns empty string for all operations.

    Supports chained attribute access (e.g. ``nonexistent.outputs.foo``)
    without raising :class:`UndefinedError`.
    """

    def __getattr__(self, name: str) -> "_PermissiveUndefined":
        return self

    def __getitem__(self, name: str) -> "_PermissiveUndefined":
        return self

    def __str__(self) -> str:
        return ""

    def __bool__(self) -> bool:
        return False

    def __iter__(self):
        return iter(())


# Use a permissive undefined that returns empty string instead of raising
_jinja_env = Environment(undefined=_PermissiveUndefined)


def _build_template_context(context: ExecutionContext) -> dict:
    """Build the Jinja2 template context from an ExecutionContext.

    Step outputs are accessible via ``steps['step_id']`` (bracket notation)
    and directly as ``step_id`` (dot notation, requires valid Python identifiers).
    Step IDs must use underscores, not hyphens, to be usable in Jinja2 dot notation.
    """
    template_ctx = {}
    template_ctx["inputs"] = context.inputs or {}
    template_ctx["steps"] = dict(context.steps)
    for step_id, step_data in context.steps.items():
        template_ctx[step_id] = step_data
    return template_ctx


def render_template(value: Any, context: ExecutionContext) -> Any:
    """Render Jinja2 templates in a value against the execution context.

    Supports str, dict (renders values), and list (renders items).
    Non-template strings pass through unchanged.
    """
    if isinstance(value, str):
        if "{{" not in value:
            return value
        try:
            template = _jinja_env.from_string(value)
            return template.render(**_build_template_context(context))
        except Exception as e:
            logger.warning(f"Template render error for '{value}': {e}")
            return value
    elif isinstance(value, dict):
        return {k: render_template(v, context) for k, v in value.items()}
    elif isinstance(value, list):
        return [render_template(v, context) for v in value]
    return value


def evaluate_condition(condition: str | None, context: ExecutionContext) -> bool:
    """Evaluate a Jinja2 condition expression against the execution context.

    Returns True if condition is None or empty (unconditional).
    Returns False on evaluation error (safe default).
    """
    if not condition:
        return True
    try:
        template = _jinja_env.from_string("{{ " + condition + " }}")
        result = template.render(**_build_template_context(context))
        # Jinja2 renders booleans as "True"/"False" strings
        if result.strip().lower() in ("true", "1", "yes"):
            return True
        if result.strip().lower() in ("false", "0", "no", "none", ""):
            return False
        return bool(result.strip())
    except Exception as e:
        logger.warning(f"Condition evaluation error for '{condition}': {e}")
        return False
