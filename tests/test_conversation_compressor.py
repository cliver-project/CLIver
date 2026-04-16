"""Tests for ConversationCompressor — conversation history compression."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cliver.config import ModelConfig  # noqa: I001
from cliver.conversation_compressor import (
    SUMMARY_PREFIX,
    ConversationCompressor,
    estimate_tokens,
    estimate_tokens_str,
    get_context_window,
)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_estimate_tokens_empty(self):
        assert estimate_tokens([]) == 0

    def test_estimate_tokens_simple(self):
        msgs = [HumanMessage(content="Hello world")]  # 11 chars -> 2 tokens
        result = estimate_tokens(msgs)
        assert result == 11 // 4

    def test_estimate_tokens_multiple_messages(self):
        msgs = [
            HumanMessage(content="a" * 100),
            AIMessage(content="b" * 200),
        ]
        result = estimate_tokens(msgs)
        assert result == 300 // 4

    def test_estimate_tokens_multipart_content(self):
        msgs = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": "data:..."},
                    {"type": "text", "text": "World"},
                ]
            )
        ]
        result = estimate_tokens(msgs)
        assert result == 10 // 4  # "Hello" + "World"

    def test_estimate_tokens_str(self):
        assert estimate_tokens_str("Hello world") == 11 // 4
        assert estimate_tokens_str("") == 0


# ---------------------------------------------------------------------------
# Context window detection
# ---------------------------------------------------------------------------


class TestGetContextWindow:
    def _model(self, name, context_window=None):
        return ModelConfig(
            name=name,
            provider="openai",
            url="http://localhost",
            context_window=context_window,
        )

    def test_explicit_context_window(self):
        m = self._model("custom-model", context_window=256000)
        assert get_context_window(m) == 256000

    def test_qwen_default(self):
        m = self._model("qwen-2.5-coder")
        assert get_context_window(m) == 131072

    def test_deepseek_default(self):
        m = self._model("deepseek-r1")
        assert get_context_window(m) == 131072

    def test_gpt4o_default(self):
        m = self._model("gpt-4o-mini")
        assert get_context_window(m) == 128000

    def test_claude_default(self):
        m = self._model("claude-sonnet-4")
        assert get_context_window(m) == 200000

    def test_unknown_fallback(self):
        m = self._model("unknown-model-xyz")
        assert get_context_window(m) == 32768

    def test_name_in_provider_used(self):
        m = ModelConfig(
            name="my-alias",
            provider="openai",
            url="http://localhost",
            name_in_provider="qwen-2.5-coder",
        )
        assert get_context_window(m) == 131072


# ---------------------------------------------------------------------------
# ConversationCompressor — needs_compression
# ---------------------------------------------------------------------------


class TestNeedsCompression:
    def test_no_compression_needed(self):
        compressor = ConversationCompressor(context_window=100000)
        history = [
            HumanMessage(content="short question"),
            AIMessage(content="short answer"),
        ]
        assert compressor.needs_compression([], history, "new question") is False

    def test_compression_needed(self):
        compressor = ConversationCompressor(context_window=100)  # tiny window
        history = [
            HumanMessage(content="a" * 200),
            AIMessage(content="b" * 200),
        ]
        assert compressor.needs_compression([], history, "new question") is True

    def test_system_messages_counted(self):
        # context_window=100, threshold=0.7 → budget=70 tokens (~280 chars)
        # system: 200 chars = 50 tokens, history: 200 chars = 50 tokens → 100 > 70
        compressor = ConversationCompressor(context_window=100)
        system = [SystemMessage(content="a" * 200)]
        history = [
            HumanMessage(content="x" * 100),
            AIMessage(content="y" * 100),
        ]
        assert compressor.needs_compression(system, history, "q") is True


# ---------------------------------------------------------------------------
# ConversationCompressor — compress
# ---------------------------------------------------------------------------


class MockLLMEngine:
    """Mock LLM engine that returns a fixed summary."""

    def __init__(self, summary="Summary of prior conversation."):
        self.summary = summary
        self.calls = []

    async def infer(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self.summary)


class FailingLLMEngine:
    """Mock LLM engine that raises an error."""

    async def infer(self, messages, tools=None, **kwargs):
        raise RuntimeError("LLM unavailable")


@pytest.mark.asyncio
class TestCompress:
    async def test_compress_returns_summary_plus_recent(self):
        compressor = ConversationCompressor(context_window=100000)
        history = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
            AIMessage(content="A2"),
            HumanMessage(content="Q3"),
            AIMessage(content="A3"),
            HumanMessage(content="Q4"),
            AIMessage(content="A4"),
        ]
        engine = MockLLMEngine()
        result = await compressor.compress(history, engine, force=True)

        # Should have a summary message + recent turns
        assert len(result) > 0
        assert isinstance(result[0], SystemMessage)
        assert result[0].content.startswith(SUMMARY_PREFIX)
        # Recent turns should be HumanMessage/AIMessage
        assert any(isinstance(m, HumanMessage) for m in result[1:])

    async def test_compress_empty_history(self):
        compressor = ConversationCompressor(context_window=100000)
        result = await compressor.compress([], MockLLMEngine())
        assert result == []

    async def test_compress_too_few_messages(self):
        compressor = ConversationCompressor(context_window=100000)
        history = [HumanMessage(content="Q1"), AIMessage(content="A1")]
        result = await compressor.compress(history, MockLLMEngine())
        # Should return original (not enough to compress)
        assert len(result) == 2

    async def test_compress_force_with_few_messages(self):
        compressor = ConversationCompressor(context_window=100000)
        history = [HumanMessage(content="Q1"), AIMessage(content="A1")]
        # Even with force, 2 messages splits at 0 so returns original
        result = await compressor.compress(history, MockLLMEngine(), force=True)
        assert len(result) == 2

    async def test_compress_fallback_on_llm_failure(self):
        # context_window=50, threshold=0.7 → budget=35 tokens (~140 chars)
        # Each message is 400 chars = 100 tokens, so truncation must drop some
        compressor = ConversationCompressor(context_window=50)
        history = [
            HumanMessage(content="Q" * 400),
            AIMessage(content="A" * 400),
            HumanMessage(content="Q" * 400),
            AIMessage(content="A" * 400),
            HumanMessage(content="Q" * 400),
            AIMessage(content="A" * 400),
        ]
        result = await compressor.compress(history, FailingLLMEngine(), force=True)

        # Should fall back to truncation — fewer messages than original
        assert len(result) < len(history)
        # First message should be truncation note
        assert isinstance(result[0], SystemMessage)
        assert "truncated" in result[0].content.lower()

    async def test_compress_calls_llm_with_older_turns(self):
        compressor = ConversationCompressor(context_window=100000, preserve_ratio=0.3)
        history = [
            HumanMessage(content=f"Q{i}")
            for i in range(10)
            for _ in (HumanMessage(content=f"Q{i}"), AIMessage(content=f"A{i}"))
        ]
        # Build proper alternating history
        history = []
        for i in range(10):
            history.append(HumanMessage(content=f"Question {i}"))
            history.append(AIMessage(content=f"Answer {i}"))

        engine = MockLLMEngine()
        await compressor.compress(history, engine, force=True)

        # LLM should have been called once with the older turns
        assert len(engine.calls) == 1


# ---------------------------------------------------------------------------
# Conversation history in AgentCore message preparation
# ---------------------------------------------------------------------------


class TestAgentCoreConversationHistory:
    """Verify conversation_history is inserted correctly into message list."""

    @pytest.mark.asyncio
    async def test_history_inserted_before_user_input(self):
        """Conversation history should appear between system messages and new user input."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from cliver.llm.llm import AgentCore

        # Create a minimal AgentCore with mocked dependencies
        model_config = ModelConfig(
            name="test-model",
            provider="openai",
            url="http://localhost:8080",
        )

        with patch("cliver.llm.llm.MCPServersCaller") as mock_mcp:
            mock_mcp_instance = MagicMock()
            mock_mcp_instance.get_mcp_tools = AsyncMock(return_value=[])
            mock_mcp.return_value = mock_mcp_instance

            executor = AgentCore(
                llm_models={"test-model": model_config},
                mcp_servers={},
                default_model="test-model",
            )

            # Mock the LLM engine
            mock_engine = MagicMock()
            mock_engine.system_message.return_value = "System prompt"
            mock_engine.config = model_config
            executor.llm_engines["test-model"] = mock_engine

            conv_history = [
                HumanMessage(content="previous question"),
                AIMessage(content="previous answer"),
            ]

            with patch("cliver.llm.llm.default_enhance_prompt", new_callable=AsyncMock, return_value=[]):
                _, _, messages = await executor._prepare_messages_and_tools(
                    enhance_prompt=None,
                    filter_tools=None,
                    model="test-model",
                    system_message_appender=None,
                    user_input="new question",
                    conversation_history=conv_history,
                )

            # Verify structure: system message(s) ... history ... new user input
            assert isinstance(messages[0], SystemMessage)  # system prompt
            # Find where history starts
            human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
            assert len(human_indices) >= 2  # at least history + new input
            # Last HumanMessage should be the new input
            assert messages[human_indices[-1]].content == "new question"
            # Previous HumanMessage should be from history
            assert messages[human_indices[-2]].content == "previous question"

    @pytest.mark.asyncio
    async def test_no_history_works_as_before(self):
        """When no conversation_history is passed, behavior is identical to before."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from cliver.llm.llm import AgentCore

        model_config = ModelConfig(
            name="test-model",
            provider="openai",
            url="http://localhost:8080",
        )

        with patch("cliver.llm.llm.MCPServersCaller") as mock_mcp:
            mock_mcp_instance = MagicMock()
            mock_mcp_instance.get_mcp_tools = AsyncMock(return_value=[])
            mock_mcp.return_value = mock_mcp_instance

            executor = AgentCore(
                llm_models={"test-model": model_config},
                mcp_servers={},
                default_model="test-model",
            )

            mock_engine = MagicMock()
            mock_engine.system_message.return_value = "System prompt"
            mock_engine.config = model_config
            executor.llm_engines["test-model"] = mock_engine

            with patch("cliver.llm.llm.default_enhance_prompt", new_callable=AsyncMock, return_value=[]):
                _, _, messages = await executor._prepare_messages_and_tools(
                    enhance_prompt=None,
                    filter_tools=None,
                    model="test-model",
                    system_message_appender=None,
                    user_input="a question",
                    conversation_history=None,
                )

            # Should have exactly 1 HumanMessage (the new input)
            human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
            assert len(human_msgs) == 1
            assert human_msgs[0].content == "a question"
