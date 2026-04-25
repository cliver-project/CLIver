"""Tests for extract_response_text — the single function for getting clean
user-visible text from any LLM response across all providers."""

from langchain_core.messages import AIMessage

from cliver.media_handler import extract_response_text


class TestNoneAndEmpty:
    def test_none_returns_fallback(self):
        assert extract_response_text(None) == ""
        assert extract_response_text(None, fallback="nope") == "nope"

    def test_empty_string_returns_fallback(self):
        assert extract_response_text("") == ""
        assert extract_response_text("", fallback="nope") == "nope"

    def test_empty_content_message(self):
        msg = AIMessage(content="")
        assert extract_response_text(msg, fallback="empty") == "empty"

    def test_none_content_message(self):
        msg = AIMessage(content="")
        msg.content = None
        assert extract_response_text(msg, fallback="none") == "none"


class TestStringContent:
    def test_plain_string(self):
        assert extract_response_text("Hello world") == "Hello world"

    def test_string_message(self):
        msg = AIMessage(content="Hello from LLM")
        assert extract_response_text(msg) == "Hello from LLM"

    def test_multiline(self):
        msg = AIMessage(content="Line 1\nLine 2\nLine 3")
        assert extract_response_text(msg) == "Line 1\nLine 2\nLine 3"

    def test_unicode_chinese(self):
        msg = AIMessage(content="你好世界")
        assert extract_response_text(msg) == "你好世界"


class TestThinkingTagStripping:
    """Models like Qwen/Ollama wrap reasoning in <think> tags."""

    def test_think_tags_removed(self):
        msg = AIMessage(content="<think>Let me reason...</think>The answer is 42.")
        assert extract_response_text(msg) == "The answer is 42."

    def test_thinking_tags_removed(self):
        msg = AIMessage(content="<thinking>reasoning here</thinking>Result is yes.")
        assert extract_response_text(msg) == "Result is yes."

    def test_only_thinking_returns_fallback(self):
        msg = AIMessage(content="<think>just reasoning, no answer</think>")
        assert extract_response_text(msg, fallback="no answer") == "no answer"

    def test_thinking_with_newlines(self):
        content = "<think>\nStep 1: ...\nStep 2: ...\n</think>\n\nFinal answer."
        msg = AIMessage(content=content)
        assert extract_response_text(msg) == "Final answer."

    def test_raw_string_with_think_tags(self):
        text = "<think>internal</think>visible"
        assert extract_response_text(text) == "visible"


class TestContentBlocksList:
    """Anthropic and similar providers return list-of-blocks."""

    def test_text_block_dict(self):
        msg = AIMessage(content=[{"type": "text", "text": "Hello"}])
        assert extract_response_text(msg) == "Hello"

    def test_thinking_block_skipped(self):
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "reasoning...", "signature": "abc123"},
                {"type": "text", "text": "The answer is 42."},
            ]
        )
        assert extract_response_text(msg) == "The answer is 42."

    def test_multiple_text_blocks_joined(self):
        msg = AIMessage(
            content=[
                {"type": "text", "text": "Part 1. "},
                {"type": "text", "text": "Part 2."},
            ]
        )
        assert extract_response_text(msg) == "Part 1. Part 2."

    def test_only_thinking_blocks_returns_fallback(self):
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "just reasoning"},
            ]
        )
        assert extract_response_text(msg, fallback="no text") == "no text"

    def test_string_items_in_list(self):
        msg = AIMessage(content=["Hello ", "world"])
        assert extract_response_text(msg) == "Hello world"

    def test_empty_list(self):
        msg = AIMessage(content=[])
        assert extract_response_text(msg, fallback="empty") == "empty"


class TestPydanticContentBlocks:
    """Anthropic LangChain integration returns Pydantic objects, not dicts.

    LangChain AIMessage validates content strictly, so we use a FakeResponse
    to simulate the real Anthropic response format with object-style blocks.
    """

    def _make_response(self, blocks):
        class FakeResponse:
            def __init__(self, content):
                self.content = content

        return FakeResponse(blocks)

    def test_pydantic_text_block(self):
        class TextBlock:
            type = "text"
            text = "Hello from Anthropic"

        resp = self._make_response([TextBlock()])
        assert extract_response_text(resp) == "Hello from Anthropic"

    def test_pydantic_thinking_block_skipped(self):
        class ThinkingBlock:
            type = "thinking"
            thinking = "deep reasoning..."
            signature = "sig_abc123"

        class TextBlock:
            type = "text"
            text = "Clean answer."

        resp = self._make_response([ThinkingBlock(), TextBlock()])
        assert extract_response_text(resp) == "Clean answer."

    def test_mixed_dict_and_pydantic(self):
        class TextBlock:
            type = "text"
            text = "from object"

        resp = self._make_response(
            [
                {"type": "thinking", "thinking": "skip me"},
                TextBlock(),
            ]
        )
        assert extract_response_text(resp) == "from object"


class TestDeepSeekReasoningContent:
    """DeepSeek puts reasoning in additional_kwargs, not in content."""

    def test_reasoning_in_kwargs_not_in_text(self):
        msg = AIMessage(
            content="The answer is 42.",
            additional_kwargs={"reasoning_content": "Let me think step by step..."},
        )
        assert extract_response_text(msg) == "The answer is 42."

    def test_reasoning_never_leaks(self):
        msg = AIMessage(
            content="Clean response.",
            additional_kwargs={"reasoning_content": "secret reasoning"},
        )
        result = extract_response_text(msg)
        assert "secret" not in result
        assert "reasoning" not in result
        assert result == "Clean response."


class TestEdgeCases:
    def test_non_message_object_with_content(self):
        class FakeResponse:
            content = "I have content"

        assert extract_response_text(FakeResponse()) == "I have content"

    def test_object_without_content(self):
        class NoContent:
            pass

        assert extract_response_text(NoContent(), fallback="nope") == "nope"

    def test_content_is_number(self):
        msg = AIMessage(content="")
        msg.content = 42
        assert extract_response_text(msg) == "42"

    def test_whitespace_only_stripped(self):
        msg = AIMessage(content="   \n\n  ")
        assert extract_response_text(msg, fallback="blank") == "blank"

    def test_thinking_then_whitespace_only(self):
        msg = AIMessage(content="<think>reasoning</think>   \n  ")
        assert extract_response_text(msg, fallback="blank") == "blank"
