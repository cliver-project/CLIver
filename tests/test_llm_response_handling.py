"""Tests for LLM response handling robustness.

Covers the hardening in base.py (arg normalization, tool call conversion),
llm.py (error detection, result formatting), and llm_utils.py (tool call parsing).
"""

from langchain_core.messages import AIMessage

from cliver.llm.llm import _format_tool_result, _is_error_response
from cliver.llm.llm_utils import (
    _coerce_json_string_values,
    _normalize_tool_args,
    normalize_tool_calls,
    parse_tool_calls_from_content,
)

# ── _normalize_tool_args ──


class TestNormalizeToolArgs:
    def test_none_returns_empty_dict(self):
        assert _normalize_tool_args(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert _normalize_tool_args("") == {}

    def test_dict_passes_through(self):
        args = {"file_path": "/tmp/test.txt", "content": "hello"}
        assert _normalize_tool_args(args) == args

    def test_json_string_parsed_to_dict(self):
        result = _normalize_tool_args('{"file_path": "/tmp/test.txt"}')
        assert result == {"file_path": "/tmp/test.txt"}

    def test_json_string_with_nested_list(self):
        """LLMs sometimes send the entire args as a JSON string."""
        json_str = '{"question": "Pick one", "options": [{"label": "A", "description": "Option A"}]}'
        result = _normalize_tool_args(json_str)
        assert result["question"] == "Pick one"
        assert isinstance(result["options"], list)
        assert result["options"][0]["label"] == "A"

    def test_non_dict_json_returns_empty(self):
        """If args string parses to a list, that's invalid — args must be a dict."""
        assert _normalize_tool_args("[1, 2, 3]") == {}

    def test_invalid_json_string_returns_empty(self):
        assert _normalize_tool_args("not json at all") == {}

    def test_unexpected_type_returns_empty(self):
        assert _normalize_tool_args(42) == {}
        assert _normalize_tool_args([1, 2]) == {}

    def test_dict_with_json_string_values_coerced(self):
        """Values inside args dict that are JSON strings should be coerced."""
        args = {
            "question": "Pick one",
            "options": '[{"label": "A", "description": "desc"}]',
        }
        result = _normalize_tool_args(args)
        assert isinstance(result["options"], list)
        assert result["options"][0]["label"] == "A"
        assert result["question"] == "Pick one"  # non-JSON string left alone


# ── _coerce_json_string_values ──


class TestCoerceJsonStringValues:
    def test_list_string_coerced(self):
        result = _coerce_json_string_values({"items": "[1, 2, 3]"})
        assert result["items"] == [1, 2, 3]

    def test_dict_string_coerced(self):
        result = _coerce_json_string_values({"config": '{"key": "val"}'})
        assert result["config"] == {"key": "val"}

    def test_plain_string_not_coerced(self):
        result = _coerce_json_string_values({"name": "hello"})
        assert result["name"] == "hello"

    def test_invalid_json_string_kept_as_is(self):
        result = _coerce_json_string_values({"data": "[not valid json"})
        assert result["data"] == "[not valid json"

    def test_non_string_values_pass_through(self):
        result = _coerce_json_string_values({"count": 5, "flag": True})
        assert result == {"count": 5, "flag": True}


# ── convert_tool_calls_for_execute (via LLMInferenceEngine) ──


class TestNormalizeToolCalls:
    """Test the normalize_tool_calls function (the normalization layer)."""

    def test_none_input(self):
        assert normalize_tool_calls(None) is None

    def test_empty_list(self):
        assert normalize_tool_calls([]) is None

    def test_missing_name_skipped(self):
        result = normalize_tool_calls(
            [
                {"args": {"x": 1}},  # no "name" key
            ]
        )
        assert result is None

    def test_none_name_skipped(self):
        result = normalize_tool_calls(
            [
                {"name": None, "args": {}},
            ]
        )
        assert result is None

    def test_valid_tool_call(self):
        result = normalize_tool_calls(
            [
                {"name": "read_file", "args": {"file_path": "/tmp/x"}, "id": "call_123"},
            ]
        )
        assert len(result) == 1
        assert result[0]["tool_name"] == "read_file"
        assert result[0]["mcp_server"] == ""
        assert result[0]["args"] == {"file_path": "/tmp/x"}
        assert result[0]["tool_call_id"] == "call_123"

    def test_mcp_tool_split(self):
        result = normalize_tool_calls(
            [
                {"name": "github#create_issue", "args": {}, "id": "c1"},
            ]
        )
        assert result[0]["mcp_server"] == "github"
        assert result[0]["tool_name"] == "create_issue"

    def test_missing_id_generates_uuid(self):
        result = normalize_tool_calls(
            [
                {"name": "read_file", "args": {}},
            ]
        )
        assert len(result[0]["tool_call_id"]) > 0

    def test_none_args_normalized(self):
        result = normalize_tool_calls(
            [
                {"name": "todo_read", "args": None, "id": "c1"},
            ]
        )
        assert result[0]["args"] == {}

    def test_string_args_parsed(self):
        result = normalize_tool_calls(
            [
                {"name": "read_file", "args": '{"file_path": "/tmp/x"}', "id": "c1"},
            ]
        )
        assert result[0]["args"] == {"file_path": "/tmp/x"}

    def test_mixed_valid_and_invalid(self):
        """Valid calls are kept, invalid ones are skipped."""
        result = normalize_tool_calls(
            [
                {"name": None, "args": {}},
                {"name": "read_file", "args": {}, "id": "c1"},
                {"args": {}},  # no name
            ]
        )
        assert len(result) == 1
        assert result[0]["tool_name"] == "read_file"


# ── _is_error_response ──


class TestIsErrorResponse:
    def test_none_is_not_error(self):
        assert _is_error_response(None) is False

    def test_normal_response_is_not_error(self):
        msg = AIMessage(content="Hello!")
        assert _is_error_response(msg) is False

    def test_error_response_detected(self):
        msg = AIMessage(content="Error: connection failed", additional_kwargs={"type": "error"})
        assert _is_error_response(msg) is True

    def test_non_error_additional_kwargs(self):
        msg = AIMessage(content="ok", additional_kwargs={"reasoning_content": "..."})
        assert _is_error_response(msg) is False


# ── _format_tool_result ──


class TestFormatToolResult:
    def test_none_result(self):
        assert _format_tool_result(None) == "(no output)"

    def test_empty_list(self):
        assert _format_tool_result([]) == "(no output)"

    def test_empty_string(self):
        assert _format_tool_result("") == "(no output)"

    def test_plain_string(self):
        assert _format_tool_result("hello world") == "hello world"

    def test_error_dict(self):
        result = _format_tool_result([{"error": "file not found"}])
        assert result == "Error: file not found"

    def test_text_dict(self):
        result = _format_tool_result([{"text": "content here"}])
        assert result == "content here"

    def test_tool_result_dict(self):
        result = _format_tool_result([{"tool_result": "done"}])
        assert result == "done"

    def test_unknown_dict_format(self):
        result = _format_tool_result([{"custom_key": "value"}])
        assert "custom_key" in result


# ── parse_tool_calls_from_content ──


class TestParseToolCallsFromContent:
    def test_none_response(self):
        assert parse_tool_calls_from_content(None) is None

    def test_structured_tool_calls(self):
        msg = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c1"}])
        result = parse_tool_calls_from_content(msg)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"

    def test_empty_content_no_tool_calls(self):
        msg = AIMessage(content="")
        assert parse_tool_calls_from_content(msg) is None

    def test_tool_calls_in_text(self):
        content = '{"tool_calls": [{"name": "read_file", "args": {"file_path": "/tmp/x"}, "id": "c1"}]}'
        msg = AIMessage(content=content)
        result = parse_tool_calls_from_content(msg)
        assert result is not None
        assert result[0]["name"] == "read_file"

    def test_tool_calls_inside_thinking_ignored(self):
        """Tool calls mentioned inside <think> blocks should NOT be parsed as real calls."""
        content = (
            '<think>I could call {"tool_calls": [{"name": "bad_tool", "args": {}}]}</think>I\'ll help you with that.'
        )
        msg = AIMessage(content=content)
        result = parse_tool_calls_from_content(msg)
        assert result is None

    def test_tool_calls_with_nested_args(self):
        """Regex-based parsing must handle nested JSON in args."""
        content = (
            '{"tool_calls": [{"name": "ask_user_question", '
            '"args": {"question": "Pick", "options": [{"label": "A", "description": "opt A"}]}, '
            '"id": "c1"}]}'
        )
        msg = AIMessage(content=content)
        result = parse_tool_calls_from_content(msg)
        assert result is not None
        assert result[0]["name"] == "ask_user_question"

    def test_no_tool_calls_keyword(self):
        msg = AIMessage(content="Just a normal response with no tools")
        assert parse_tool_calls_from_content(msg) is None
