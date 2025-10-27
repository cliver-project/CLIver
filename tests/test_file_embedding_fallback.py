"""
Test module for file embedding fallback functionality in CLIver.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage

from cliver.config import ModelConfig, ModelOptions
from cliver.llm.base import LLMInferenceEngine
from cliver.llm.llm import TaskExecutor
from cliver.model_capabilities import ModelCapability


class TestFileEmbeddingFallback:
    """Test file embedding fallback functionality."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock LLM engine without file upload support."""
        config = ModelConfig(
            name="test-model",
            provider="test",
            name_in_provider="test-model",
            url="http://test",
            api_key=None,
            options=ModelOptions(
                temperature=0.5,
            ),
        )

        # Create a mock engine
        engine = Mock(spec=LLMInferenceEngine)
        engine.config = config

        # Mock capabilities (no file upload support)
        engine.config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        # Mock other required methods
        engine.system_message = Mock(return_value="You are a helpful assistant.")
        engine.infer = AsyncMock(return_value=AIMessage(content="Test response"))
        engine.parse_tool_calls = Mock(return_value=None)

        return engine

    @pytest.fixture
    def task_executor(self, mock_engine):
        """Create a mock TaskExecutor."""
        llm_models = {"test-model": mock_engine.config}
        mcp_servers = {}
        executor = TaskExecutor(llm_models, mcp_servers)
        executor.llm_engines["test-model"] = mock_engine
        return executor

    def test_process_user_input_with_files_fallback(self, task_executor, mock_engine):
        """Test processing user input with file embedding fallback."""
        # Mock the LLM response
        mock_response = AIMessage(content="I've analyzed the file contents.")
        mock_engine.infer = AsyncMock(return_value=mock_response)

        # Create a temporary test file
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file content for embedding.")
            test_file_path = f.name

        try:
            # Process user input with files (should use embedding fallback)
            response = task_executor.process_user_input_sync(
                user_input="Analyze this file",
                files=[test_file_path],
                model="test-model",
            )

            # Verify the response
            assert response.content == "I've analyzed the file contents."

            # Verify that infer was called with messages containing embedded content
            mock_engine.infer.assert_called_once()
            args, kwargs = mock_engine.infer.call_args
            messages = args[0]  # First argument should be messages

            # Check that the last message contains embedded file content
            last_message = messages[-1]
            assert hasattr(last_message, "content")
            assert isinstance(last_message.content, list)

            # Check for text content part
            text_parts = [part for part in last_message.content if part.get("type") == "text"]
            assert len(text_parts) >= 2  # User input + embedded file header

            # Check that file content is embedded
            embedded_content = "".join([part.get("text", "") for part in text_parts])
            assert "This is a test file content for embedding." in embedded_content

        finally:
            # Clean up the temporary file
            os.unlink(test_file_path)

    def test_process_user_input_with_multiple_files_fallback(self, task_executor, mock_engine):
        """Test processing user input with multiple files using embedding fallback."""
        # Mock the LLM response
        mock_response = AIMessage(content="I've analyzed the file contents.")
        mock_engine.infer = AsyncMock(return_value=mock_response)

        # Create temporary test files
        import os
        import tempfile

        test_files = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{i}.txt", delete=False) as f:
                    f.write(f"This is test file {i} content for embedding.")
                    test_files.append(f.name)

            # Process user input with multiple files (should use embedding fallback)
            response = task_executor.process_user_input_sync(
                user_input="Analyze these files", files=test_files, model="test-model"
            )

            # Verify the response
            assert response.content == "I've analyzed the file contents."

            # Verify that infer was called with messages containing embedded content
            mock_engine.infer.assert_called_once()
            args, kwargs = mock_engine.infer.call_args
            messages = args[0]  # First argument should be messages

            # Check that the last message contains embedded file content
            last_message = messages[-1]
            assert hasattr(last_message, "content")
            assert isinstance(last_message.content, list)

            # Check for text content part
            text_parts = [part for part in last_message.content if part.get("type") == "text"]
            assert len(text_parts) >= 3  # User input + embedded files header + 2 files

            # Check that file content is embedded
            embedded_content = "".join([part.get("text", "") for part in text_parts])
            assert "This is test file 0 content for embedding." in embedded_content
            assert "This is test file 1 content for embedding." in embedded_content

        finally:
            # Clean up the temporary files
            for file_path in test_files:
                os.unlink(file_path)


if __name__ == "__main__":
    pytest.main([__file__])
