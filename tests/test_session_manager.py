"""Tests for SessionManager — SQLite-backed session CRUD, recording, and search."""

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
        assert info["title"] is None or info["title"] == ""

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
        manager.append_turn(s1, "user", "update")
        sessions = manager.list_sessions()
        assert sessions[0]["id"] == s1

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

    def test_turn_count_updates(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "q1")
        manager.append_turn(sid, "assistant", "a1")
        manager.append_turn(sid, "user", "q2")

        info = manager.get_session_info(sid)
        assert info["turn_count"] == 3

    def test_title_set_from_first_user_message(self, manager):
        sid = manager.create_session()
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
# Options persistence
# ---------------------------------------------------------------------------


class TestOptions:
    def test_save_and_load_options(self, manager):
        sid = manager.create_session()
        manager.save_options(sid, {"model": "qwen", "temperature": 0.7})
        opts = manager.load_options(sid)
        assert opts["model"] == "qwen"
        assert opts["temperature"] == 0.7

    def test_load_options_empty(self, manager):
        sid = manager.create_session()
        assert manager.load_options(sid) == {}

    def test_load_options_nonexistent(self, manager):
        assert manager.load_options("nope") == {}


# ---------------------------------------------------------------------------
# Unicode
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


# ---------------------------------------------------------------------------
# Full-text search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_matching_content(self, manager):
        sid = manager.create_session("Kubernetes help")
        manager.append_turn(sid, "user", "How do I deploy to kubernetes?")
        manager.append_turn(sid, "assistant", "Use kubectl apply to deploy your manifests.")

        results = manager.search("kubernetes")
        assert len(results) == 1
        assert results[0]["session_id"] == sid
        assert len(results[0]["snippets"]) >= 1

    def test_search_no_results(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "Hello world")
        results = manager.search("kubernetes")
        assert results == []

    def test_search_multiple_sessions(self, manager):
        s1 = manager.create_session("Python help")
        manager.append_turn(s1, "user", "How to use python decorators?")
        s2 = manager.create_session("Also python")
        manager.append_turn(s2, "user", "Python async programming")
        s3 = manager.create_session("Java stuff")
        manager.append_turn(s3, "user", "Java spring boot setup")

        results = manager.search("python")
        assert len(results) == 2
        session_ids = {r["session_id"] for r in results}
        assert s1 in session_ids
        assert s2 in session_ids
        assert s3 not in session_ids

    def test_search_chinese(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "如何部署到 kubernetes 集群？")
        results = manager.search("kubernetes")
        assert len(results) == 1

    def test_search_returns_snippets(self, manager):
        sid = manager.create_session()
        manager.append_turn(sid, "user", "Tell me about kubernetes deployment strategies")
        results = manager.search("kubernetes")
        assert len(results) == 1
        assert len(results[0]["snippets"]) >= 1
        snippet = results[0]["snippets"][0]
        assert "role" in snippet
        assert "content" in snippet

    def test_search_respects_limit(self, manager):
        for i in range(10):
            sid = manager.create_session()
            manager.append_turn(sid, "user", f"Python question {i}")
        results = manager.search("python", limit=3)
        assert len(results) <= 3

    def test_search_with_title(self, manager):
        sid = manager.create_session("My deployment guide")
        manager.append_turn(sid, "user", "kubernetes setup steps")
        results = manager.search("kubernetes")
        assert results[0]["title"] == "My deployment guide"

    def test_update_title(self, manager):
        sid = manager.create_session()
        manager.update_title(sid, "New title")
        info = manager.get_session_info(sid)
        assert info["title"] == "New title"
