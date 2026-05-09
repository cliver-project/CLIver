# tests/test_node_runners.py

import pytest

from cliver.workflow.node_runners import run_python_node


class TestRunPythonNode:
    def test_basic_python_node(self, tmp_path):
        script = tmp_path / "transform.py"
        script.write_text("def run(inputs):\n    return {'count': len(inputs.get('items', []))}\n")
        result = run_python_node(str(script), {"items": [1, 2, 3]})
        assert result == {"count": 3}

    def test_python_node_receives_step_outputs(self, tmp_path):
        script = tmp_path / "merge.py"
        script.write_text("def run(inputs):\n    return {'merged': inputs['a'] + inputs['b']}\n")
        result = run_python_node(str(script), {"a": "hello", "b": " world"})
        assert result == {"merged": "hello world"}

    def test_python_node_missing_run_function(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("x = 1\n")
        with pytest.raises(AttributeError, match="run"):
            run_python_node(str(script), {})

    def test_python_node_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_python_node("/nonexistent/script.py", {})
