# tests/test_condition_eval.py
from cliver.workflow.condition_eval import evaluate_condition


class TestEvaluateCondition:
    def test_none_condition_returns_true(self):
        assert evaluate_condition(None, {}) is True

    def test_empty_condition_returns_true(self):
        assert evaluate_condition("", {}) is True

    def test_equality_string(self):
        steps = {"research": {"sentiment": "positive"}}
        assert evaluate_condition("research.sentiment == 'positive'", steps) is True
        assert evaluate_condition("research.sentiment == 'negative'", steps) is False

    def test_equality_number(self):
        steps = {"s1": {"count": 5}}
        assert evaluate_condition("s1.count == 5", steps) is True
        assert evaluate_condition("s1.count == 3", steps) is False

    def test_inequality_gt(self):
        steps = {"s1": {"count": 10}}
        assert evaluate_condition("s1.count > 5", steps) is True
        assert evaluate_condition("s1.count > 15", steps) is False

    def test_inequality_lt(self):
        steps = {"s1": {"count": 3}}
        assert evaluate_condition("s1.count < 5", steps) is True

    def test_inequality_gte_lte(self):
        steps = {"s1": {"count": 5}}
        assert evaluate_condition("s1.count >= 5", steps) is True
        assert evaluate_condition("s1.count <= 5", steps) is True
        assert evaluate_condition("s1.count >= 6", steps) is False

    def test_not_equal(self):
        steps = {"s1": {"status": "ok"}}
        assert evaluate_condition("s1.status != 'error'", steps) is True
        assert evaluate_condition("s1.status != 'ok'", steps) is False

    def test_boolean_truthy(self):
        steps = {"s1": {"success": True}}
        assert evaluate_condition("s1.success", steps) is True

    def test_boolean_falsy(self):
        steps = {"s1": {"success": False}}
        assert evaluate_condition("s1.success", steps) is False

    def test_not_operator(self):
        steps = {"s1": {"error": False}}
        assert evaluate_condition("not s1.error", steps) is True

    def test_and_operator(self):
        steps = {"s1": {"done": True}, "s2": {"done": True}}
        assert evaluate_condition("s1.done and s2.done", steps) is True
        steps["s2"]["done"] = False
        assert evaluate_condition("s1.done and s2.done", steps) is False

    def test_or_operator(self):
        steps = {"s1": {"done": False}, "s2": {"done": True}}
        assert evaluate_condition("s1.done or s2.done", steps) is True

    def test_missing_path_is_falsy(self):
        assert evaluate_condition("missing.key", {}) is False

    def test_nested_path(self):
        steps = {"s1": {"data": {"status": "ok"}}}
        assert evaluate_condition("s1.data.status == 'ok'", steps) is True
