from unittest.mock import ANY, AsyncMock, Mock, patch

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
        with patch("cliver.tools.image_generate.get_task_executor", return_value=None):
            result = tool._run(prompt="a cat")
            assert "error" in result.lower()

    def test_run_success(self):
        tool = ImageGenerateTool()
        mock_executor = Mock()
        mock_media = [MediaContent(type=MediaType.IMAGE, data="https://img.png", mime_type="image/png", source="test")]
        mock_result = AIMessage(
            content="Generated 1 image(s):\nhttps://img.png",
            additional_kwargs={"media_content": mock_media},
        )
        mock_executor.generate_image = AsyncMock(return_value=mock_result)

        with patch("cliver.tools.image_generate.get_task_executor", return_value=mock_executor):
            result = tool._run(prompt="a cat")
            assert "https://img.png" in result
            mock_executor.generate_image.assert_called_once_with("a cat", None, ctx=ANY)

    def test_run_with_model(self):
        tool = ImageGenerateTool()
        mock_executor = Mock()
        mock_result = AIMessage(content="Generated 1 image(s):\nhttps://img.png")
        mock_executor.generate_image = AsyncMock(return_value=mock_result)

        with patch("cliver.tools.image_generate.get_task_executor", return_value=mock_executor):
            tool._run(prompt="a dog", model="minimax-image")
            mock_executor.generate_image.assert_called_once_with("a dog", "minimax-image", ctx=ANY)

    def test_run_handles_exception(self):
        tool = ImageGenerateTool()
        mock_executor = Mock()
        mock_executor.generate_image = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("cliver.tools.image_generate.get_task_executor", return_value=mock_executor):
            result = tool._run(prompt="a cat")
            assert "error" in result.lower()
            assert "API down" in result
