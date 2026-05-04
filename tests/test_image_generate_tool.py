from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage

from cliver.media import MediaContent, MediaType
from cliver.tools.image_generate import ImageGenerateTool


class TestImageGenerateTool:
    def test_tool_metadata(self):
        tool = ImageGenerateTool()
        assert tool.name == "ImageGenerate"
        assert "image" in tool.description.lower()

    def test_run_no_executor(self):
        tool = ImageGenerateTool()
        with patch("cliver.tools.image_generate.get_agent_core", return_value=None):
            result = tool._run(prompt="a cat")
            assert "error" in result.lower()

    def test_run_success_url(self):
        """URL images are returned as URLs without downloading."""
        tool = ImageGenerateTool()
        mock_executor = Mock()
        mock_media = MediaContent(type=MediaType.IMAGE, data="https://img.png", mime_type="image/png", source="url")
        mock_result = AIMessage(content="Generated 1 image(s).")

        def mock_generate(prompt, ctx):
            ctx.generated_media.append(mock_media)
            return mock_result

        mock_executor.generate_image = AsyncMock(side_effect=mock_generate)

        with patch("cliver.tools.image_generate.get_agent_core", return_value=mock_executor):
            result = tool._run(prompt="a cat")
            assert "https://img.png" in result
            assert "Generated 1 image" in result

    def test_run_handles_exception(self):
        tool = ImageGenerateTool()
        mock_executor = Mock()
        mock_executor.generate_image = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("cliver.tools.image_generate.get_agent_core", return_value=mock_executor):
            result = tool._run(prompt="a cat")
            assert "error" in result.lower()
            assert "API down" in result
