"""Safe dot-path expression evaluator for workflow conditions.

Supports: dot-path access, ==, !=, >, <, >=, <=, and, or, not.
No arbitrary code execution, no Jinja2 — just structured parsing.
"""

import re
from typing import Any, Dict, Optional


def _traverse(path: str, steps: Dict[str, Any]) -> Any:
    """Walk a dot-path into the steps dict. Returns None if missing."""
    parts = path.split(".")
    current: Any = steps
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


_COMPARISON_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}

_COMPARISON_RE = re.compile(r"^(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+)$")


def _parse_literal(s: str) -> Any:
    """Parse a string/number/bool literal."""
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("None", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _eval_atom(expr: str, steps: Dict[str, Any]) -> Any:
    """Evaluate a single atom: either a comparison or a dot-path lookup."""
    expr = expr.strip()

    if expr.startswith("not "):
        inner = expr[4:].strip()
        return not _eval_atom(inner, steps)

    m = _COMPARISON_RE.match(expr)
    if m:
        left_raw, op, right_raw = m.group(1).strip(), m.group(2), m.group(3).strip()
        left_val = _traverse(left_raw, steps)
        if left_val is None and not left_raw.startswith(("'", '"')):
            left_val = _parse_literal(left_raw)
        right_val = _parse_literal(right_raw)
        if left_val is None:
            return False
        return _COMPARISON_OPS[op](left_val, right_val)

    value = _traverse(expr, steps)
    return bool(value) if value is not None else False


def evaluate_condition(condition: Optional[str], steps: Dict[str, Any]) -> bool:
    """Evaluate a condition expression against step outputs.

    Returns True if condition is None or empty (unconditional step).
    """
    if not condition:
        return True

    condition = condition.strip()

    if " or " in condition:
        parts = condition.split(" or ")
        return any(_eval_atom(p, steps) for p in parts)

    if " and " in condition:
        parts = condition.split(" and ")
        return all(_eval_atom(p, steps) for p in parts)

    return _eval_atom(condition, steps)
