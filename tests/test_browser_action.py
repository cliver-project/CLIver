"""Tests for browser_action tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from cliver.tools.browser_action import BrowserActionTool, BrowserSession


class TestBrowserActionTool:
    def test_tool_name(self):
        tool = BrowserActionTool()
        assert tool.name == "browser_action"

    def test_unknown_action(self):
        tool = BrowserActionTool()
        with patch("cliver.tools.browser_action.get_browser_session") as mock_get:
            mock_session = MagicMock()
            mock_page = AsyncMock()
            mock_session.ensure_started = AsyncMock(return_value=mock_page)
            mock_get.return_value = mock_session

            result = asyncio.run(tool._async_run("unknown_action"))
            assert "Unknown action" in result

    def test_navigate_requires_value(self):
        tool = BrowserActionTool()
        with patch("cliver.tools.browser_action.get_browser_session") as mock_get:
            mock_session = MagicMock()
            mock_page = AsyncMock()
            mock_session.ensure_started = AsyncMock(return_value=mock_page)
            mock_get.return_value = mock_session

            result = asyncio.run(tool._async_run("navigate"))
            assert "required" in result.lower()

    def test_click_requires_selector(self):
        tool = BrowserActionTool()
        with patch("cliver.tools.browser_action.get_browser_session") as mock_get:
            mock_session = MagicMock()
            mock_page = AsyncMock()
            mock_session.ensure_started = AsyncMock(return_value=mock_page)
            mock_get.return_value = mock_session

            result = asyncio.run(tool._async_run("click"))
            assert "required" in result.lower()

    def test_fill_requires_selector(self):
        tool = BrowserActionTool()
        with patch("cliver.tools.browser_action.get_browser_session") as mock_get:
            mock_session = MagicMock()
            mock_page = AsyncMock()
            mock_session.ensure_started = AsyncMock(return_value=mock_page)
            mock_get.return_value = mock_session

            result = asyncio.run(tool._async_run("fill"))
            assert "required" in result.lower()

    def test_evaluate_requires_value(self):
        tool = BrowserActionTool()
        with patch("cliver.tools.browser_action.get_browser_session") as mock_get:
            mock_session = MagicMock()
            mock_page = AsyncMock()
            mock_session.ensure_started = AsyncMock(return_value=mock_page)
            mock_get.return_value = mock_session

            result = asyncio.run(tool._async_run("evaluate"))
            assert "required" in result.lower()

    def test_tool_registered(self):
        from cliver.tool_registry import ToolRegistry

        registry = ToolRegistry()
        # May be excluded from LLM tools (no playwright) but always accessible by name
        tool = registry.get_tool_by_name("browser_action")
        assert tool is not None


class TestBrowserSession:
    def test_initial_state(self):
        session = BrowserSession()
        assert session.is_active is False

    def test_close_when_not_started(self):
        session = BrowserSession()
        asyncio.run(session.close())  # should not raise
        assert session.is_active is False
