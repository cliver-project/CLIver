"""
Test module for file upload functionality in CLIver.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from cliver.llm.llm import TaskExecutor
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
from cliver.config import ModelConfig
from cliver.model_capabilities import ModelCapability
from langchain_core.messages import AIMessage


class TestFileUpload:
    """Test file upload functionality."""

    @pytest.fixture
    def openai_engine(self):
        """Create a mock OpenAI engine with file upload support."""
        config = Mock(spec=ModelConfig)
        config.name = "gpt-4"
        config.provider = "openai"
        config.name_in_provider = "gpt-4"
        config.url = "https://api.openai.com/v1"
        config.api_key = "test-key"
        config.options = None

        # Mock capabilities
        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.TOOL_CALLING,
                ModelCapability.FILE_UPLOAD,
            }
        )

        engine = OpenAICompatibleInferenceEngine(config)
        return engine

    @pytest.fixture
    def task_executor(self, openai_engine):
        """Create a mock TaskExecutor."""
        llm_models = {"gpt-4": openai_engine.config}
        mcp_servers = {}
        executor = TaskExecutor(llm_models, mcp_servers)
        executor.llm_engines["gpt-4"] = openai_engine
        return executor

    def test_process_user_input_with_files(self, task_executor, openai_engine):
        """Test processing user input with file uploads."""
        # Mock the LLM response
        mock_response = AIMessage(content="I've analyzed the files.")
        openai_engine.infer = AsyncMock(return_value=mock_response)

        # Mock the file upload
        openai_engine.upload_file = Mock(return_value="file-12345")

        # Process user input with files
        response = task_executor.process_user_input_sync(
            user_input="Analyze these files",
            files=["test.txt", "data.csv"],
            model="gpt-4"
        )

        # Verify the response
        assert response.content == "I've analyzed the files."

        # Verify file upload was called
        assert openai_engine.upload_file.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])