"""Tests for SessionManager — conversation session CRUD and recording."""

import pytest

from cliver.session_manager import SessionManager


@pytest.fixture
def manager(tmp_path):
    return SessionManager(tmp_path / "sessions")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_create_session(self, manager):
        sid = manager.create_session("Hello world")
        assert len(sid) == 8
        info = manager.get_session_info(sid)
        assert info is not None
        assert info["title"] == "Hello world"
        assert info["turn_count"] == 0

    def test_create_without_title(self, manager):
        sid = manager.create_session()
        info = manager.get_session_info(sid)
        assert info["title"] == ""

    def test_list_sessions_empty(self, manager):
        assert manager.list_sessions() == []

    def test_list_sessions(self, manager):
        manager.create_session("First")
        manager.create_session("Second")
        sessions = manager.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_most_recent_first(self, manager):
        s1 = manager.create_session("Older")
        manager.create_session("Newer")
        # Append to s1 to make it more recently updated
        manager.append_turn(s1, "user", "update")
        sessions = manager.list_sessions()
        assert sessions[0]["id"] == s1  # s1 was updated more recently

    def test_delete_session(self, manager):
        sid = manager.create_session("To delete")
        assert manager.delete_session(sid) is True
        assert manager.get_session_info(sid) is None
        assert manager.list_sessions() == []

    def test_delete_nonexistent(self, manager):
        assert manager.delete_session("nope") is False

    def test_get_nonexistent(self, manager):
        assert manager.get_session_info("nope") is None


# ---------------------------------------------------------------------------
# Conversation recording
# ---------------------------------------------------------------------------


class TestConversationRecording:
    def test_append_and_load_turns(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "Hello")
        manager.append_turn(sid, "assistant", "Hi! How can I help?")

        turns = manager.load_turns(sid)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hi! How can I help?"

    def test_turns_have_timestamps(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "test")
        turns = manager.load_turns(sid)
        assert "timestamp" in turns[0]
        assert "UTC" in turns[0]["timestamp"]

    def test_turn_count_updates(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "q1")
        manager.append_turn(sid, "assistant", "a1")
        manager.append_turn(sid, "user", "q2")

        info = manager.get_session_info(sid)
        assert info["turn_count"] == 3

    def test_title_set_from_first_user_message(self, manager):
        sid = manager.create_session()  # no title
        manager.append_turn(sid, "user", "What is the meaning of life?")

        info = manager.get_session_info(sid)
        assert info["title"] == "What is the meaning of life?"

    def test_title_not_overwritten(self, manager):
        sid = manager.create_session("Original title")
        manager.append_turn(sid, "user", "Should not replace title")

        info = manager.get_session_info(sid)
        assert info["title"] == "Original title"

    def test_title_truncated_at_80_chars(self, manager):
        sid = manager.create_session()
        long_msg = "x" * 200
        manager.append_turn(sid, "user", long_msg)

        info = manager.get_session_info(sid)
        assert len(info["title"]) == 80

    def test_load_empty_session(self, manager):
        sid = manager.create_session()
        turns = manager.load_turns(sid)
        assert turns == []

    def test_load_nonexistent_session(self, manager):
        turns = manager.load_turns("nonexistent")
        assert turns == []

    def test_multiple_sessions_isolated(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()

        manager.append_turn(s1, "user", "Session 1 message")
        manager.append_turn(s2, "user", "Session 2 message")

        t1 = manager.load_turns(s1)
        t2 = manager.load_turns(s2)

        assert len(t1) == 1
        assert t1[0]["content"] == "Session 1 message"
        assert len(t2) == 1
        assert t2[0]["content"] == "Session 2 message"

    def test_delete_removes_turns(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "test")
        manager.delete_session(sid)
        assert manager.load_turns(sid) == []


# ---------------------------------------------------------------------------
# Unicode and special characters
# ---------------------------------------------------------------------------


class TestUnicode:
    def test_unicode_content(self, manager):
        sid = manager.create_session("日本語テスト")
        manager.append_turn(sid, "user", "こんにちは世界")
        manager.append_turn(sid, "assistant", "你好世界 🌍")

        turns = manager.load_turns(sid)
        assert turns[0]["content"] == "こんにちは世界"
        assert turns[1]["content"] == "你好世界 🌍"

        info = manager.get_session_info(sid)
        assert info["title"] == "日本語テスト"
