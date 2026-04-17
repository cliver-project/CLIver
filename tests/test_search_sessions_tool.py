"""Tests for the search_sessions builtin tool."""

from unittest.mock import MagicMock, patch

from cliver.tools.search_sessions import SearchSessionsTool


class TestSearchSessionsTool:
    def test_tool_has_correct_name(self):
        tool = SearchSessionsTool()
        assert tool.name == "SearchSessions"

    def test_tool_has_description(self):
        tool = SearchSessionsTool()
        assert len(tool.description) > 20

    @patch("cliver.tools.search_sessions.get_current_profile")
    def test_search_returns_results(self, mock_profile):
        mock_sm = MagicMock()
        mock_sm.search.return_value = [
            {
                "session_id": "abc12345",
                "title": "Kubernetes help",
                "created_at": "2026-04-14 10:00 UTC",
                "snippets": [{"role": "user", "content": "How to deploy?"}],
            }
        ]
        mock_agent = MagicMock()
        mock_agent.sessions_dir = "/tmp/sessions"
        mock_profile.return_value = mock_agent

        tool = SearchSessionsTool()
        with patch("cliver.session_manager.SessionManager") as MockSM:
            MockSM.return_value = mock_sm
            result = tool._run(query="kubernetes")

        assert "abc12345" in result
        assert "Kubernetes help" in result

    @patch("cliver.tools.search_sessions.get_current_profile")
    def test_search_no_profile(self, mock_profile):
        mock_profile.return_value = None
        tool = SearchSessionsTool()
        result = tool._run(query="test")
        assert "not available" in result.lower()

    @patch("cliver.tools.search_sessions.get_current_profile")
    def test_search_no_results(self, mock_profile):
        mock_sm = MagicMock()
        mock_sm.search.return_value = []
        mock_agent = MagicMock()
        mock_agent.sessions_dir = "/tmp/sessions"
        mock_profile.return_value = mock_agent

        tool = SearchSessionsTool()
        with patch("cliver.session_manager.SessionManager") as MockSM:
            MockSM.return_value = mock_sm
            result = tool._run(query="nonexistent")

        assert "no" in result.lower()
