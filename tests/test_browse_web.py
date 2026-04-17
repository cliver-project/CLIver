"""Tests for browse_web tool."""

from unittest.mock import MagicMock, patch

from cliver.tools.browse_web import BrowseWebTool


class TestBrowseWebTool:
    def test_tool_name(self):
        tool = BrowseWebTool()
        assert tool.name == "Browse"

    @patch.dict("os.environ", {"FIRECRAWL_API_KEY": ""}, clear=False)
    def test_missing_api_key(self):
        tool = BrowseWebTool()
        result = tool._run(url="https://example.com")
        assert "FIRECRAWL_API_KEY" in result

    @patch.dict("os.environ", {"FIRECRAWL_API_KEY": "test-key"})
    def test_scrape_success(self):
        import sys

        # Create a mock firecrawl module so the local import succeeds
        mock_firecrawl = MagicMock()
        mock_app = MagicMock()
        mock_app.scrape_url.return_value = {"markdown": "# Hello World"}
        mock_firecrawl.FirecrawlApp.return_value = mock_app

        with patch.dict(sys.modules, {"firecrawl": mock_firecrawl}):
            tool = BrowseWebTool()
            result = tool._run(url="https://example.com")

        assert "Hello World" in result

    def test_tool_accessible_by_name(self):
        """browse_web may be excluded from LLM tools (no API key) but always accessible by name."""
        from cliver.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tool = registry.get_tool_by_name("Browse")
        assert tool is not None
