# tests/test_node_runners.py

import pytest

from cliver.workflow.node_runners import run_python_node


class TestRunPythonNode:
    def test_basic_python_node(self, tmp_path):
        script = tmp_path / "transform.py"
        script.write_text("def run(inputs):\n    return {'count': len(inputs.get('items', []))}\n")
        result = run_python_node({"items": [1, 2, 3]}, file_path=str(script))
        assert result == {"count": 3}

    def test_python_node_receives_step_outputs(self, tmp_path):
        script = tmp_path / "merge.py"
        script.write_text("def run(inputs):\n    return {'merged': inputs['a'] + inputs['b']}\n")
        result = run_python_node({"a": "hello", "b": " world"}, file_path=str(script))
        assert result == {"merged": "hello world"}

    def test_python_node_missing_run_function(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("x = 1\n")
        with pytest.raises(AttributeError, match="run"):
            run_python_node({}, file_path=str(script))

    def test_python_node_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_python_node({}, file_path="/nonexistent/script.py")

    def test_inline_code(self):
        code = "def run(inputs):\n    return {'sum': inputs['a'] + inputs['b']}\n"
        result = run_python_node({"a": 3, "b": 4}, code=code)
        assert result == {"sum": 7}

    def test_inline_code_missing_run(self):
        with pytest.raises(AttributeError, match="run"):
            run_python_node({}, code="x = 1")

    def test_neither_file_nor_code_raises(self):
        with pytest.raises(ValueError, match="file_path or code"):
            run_python_node({})

    def test_two_arg_run_receives_state(self):
        code = "def run(inputs, state):\n    return {'dir': state.get('outputs_dir', 'none')}\n"
        result = run_python_node({}, code=code, state={"outputs_dir": "/tmp/out"})
        assert result == {"dir": "/tmp/out"}

    def test_one_arg_backward_compat(self):
        code = "def run(inputs):\n    return {'ok': True}\n"
        result = run_python_node({}, code=code, state={"outputs_dir": "/tmp"})
        assert result == {"ok": True}
