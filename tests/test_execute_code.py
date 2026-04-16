"""Tests for execute_code sandbox tool."""

from cliver.tools.execute_code import ExecuteCodeTool


class TestExecuteCodeTool:
    def test_tool_name(self):
        tool = ExecuteCodeTool()
        assert tool.name == "execute_code"

    def test_simple_print(self):
        tool = ExecuteCodeTool()
        result = tool._run(code='print("hello world")')
        assert "hello world" in result

    def test_math_calculation(self):
        tool = ExecuteCodeTool()
        result = tool._run(code="print(2 + 2)")
        assert "4" in result

    def test_multiline_script(self):
        tool = ExecuteCodeTool()
        result = tool._run(code="""
import json
data = {"name": "CLIver", "tools": 20}
print(json.dumps(data))
""")
        assert "CLIver" in result
        assert "20" in result

    def test_file_read_write(self, tmp_path, monkeypatch):
        """Script can read/write files in cwd."""
        monkeypatch.chdir(tmp_path)
        # Create a file to read
        (tmp_path / "input.txt").write_text("hello from file")

        tool = ExecuteCodeTool()
        result = tool._run(code="""
with open("input.txt") as f:
    content = f.read()
with open("output.txt", "w") as f:
    f.write(content.upper())
print("done")
""")
        assert "done" in result
        assert (tmp_path / "output.txt").read_text() == "HELLO FROM FILE"

    def test_stderr_captured(self):
        tool = ExecuteCodeTool()
        result = tool._run(code='import sys; sys.stderr.write("warning\\n")')
        assert "warning" in result
        assert "STDERR" in result

    def test_syntax_error_reported(self):
        tool = ExecuteCodeTool()
        result = tool._run(code="def oops(")
        assert "SyntaxError" in result or "Exit code" in result

    def test_exception_reported(self):
        tool = ExecuteCodeTool()
        result = tool._run(code='raise ValueError("test error")')
        assert "ValueError" in result or "test error" in result

    def test_timeout(self):
        tool = ExecuteCodeTool()
        result = tool._run(code="import time; time.sleep(10)", timeout=1)
        assert "timed out" in result.lower()

    def test_no_output(self):
        tool = ExecuteCodeTool()
        result = tool._run(code="x = 42")
        assert "no output" in result.lower()

    def test_output_truncation(self):
        tool = ExecuteCodeTool()
        result = tool._run(code='print("x" * 100000)')
        assert "truncated" in result

    def test_tool_in_core_toolset(self):
        from cliver.tool_registry import TOOLSETS
        assert "execute_code" in TOOLSETS["core"]

    def test_tool_registered(self):
        from cliver.tool_registry import ToolRegistry
        registry = ToolRegistry()
        assert "execute_code" in registry.tool_names
