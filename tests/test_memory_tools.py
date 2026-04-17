"""Tests for memory_read, memory_write, and identity_update builtin tools."""

import pytest

import cliver.agent_profile as ap
from cliver.agent_profile import AgentProfile
from cliver.tools.memory import IdentityUpdateTool, MemoryReadTool, MemoryWriteTool


@pytest.fixture
def profile(tmp_path):
    """Create an AgentProfile with a temp config directory."""
    return AgentProfile("TestBot", config_dir=tmp_path)


@pytest.fixture(autouse=True)
def wire_profile(profile, monkeypatch):
    """Set the test profile as the current profile via the centralized registry."""
    monkeypatch.setattr(ap, "_current_profile", profile)
    yield
    monkeypatch.setattr(ap, "_current_profile", None)


# ---------------------------------------------------------------------------
# memory_read
# ---------------------------------------------------------------------------


class TestMemoryRead:
    def test_empty_memory(self):
        tool = MemoryReadTool()
        result = tool._run()
        assert "No memories" in result

    def test_reads_after_write(self, profile):
        profile.append_memory("User prefers dark mode")
        tool = MemoryReadTool()
        result = tool._run()
        assert "dark mode" in result

    def test_reads_global_and_agent(self, profile):
        profile.append_memory("Global fact", scope="global")
        profile.append_memory("Agent fact", scope="agent")
        tool = MemoryReadTool()
        result = tool._run()
        assert "Global fact" in result
        assert "Agent fact" in result

    def test_no_profile_returns_message(self, monkeypatch):
        monkeypatch.setattr(ap, "_current_profile", None)
        tool = MemoryReadTool()
        result = tool._run()
        assert "not available" in result


# ---------------------------------------------------------------------------
# memory_write
# ---------------------------------------------------------------------------


class TestMemoryWrite:
    def test_write_agent_memory(self, profile):
        tool = MemoryWriteTool()
        result = tool._run(content="User uses uv, not pip")
        assert "Saved to agent memory" in result

        content = profile.memory_file.read_text()
        assert "User uses uv, not pip" in content

    def test_write_global_memory(self, profile, tmp_path):
        tool = MemoryWriteTool()
        result = tool._run(content="Shared knowledge", scope="global")
        assert "Saved to global memory" in result

        content = (tmp_path / "memory.md").read_text()
        assert "Shared knowledge" in content

    def test_default_scope_is_agent(self, profile):
        tool = MemoryWriteTool()
        tool._run(content="Should be agent-scoped")

        assert profile.memory_file.exists()
        content = profile.memory_file.read_text()
        assert "Should be agent-scoped" in content

    def test_no_profile_returns_message(self, monkeypatch):
        monkeypatch.setattr(ap, "_current_profile", None)
        tool = MemoryWriteTool()
        result = tool._run(content="test")
        assert "not available" in result

    def test_multiple_writes_are_additive(self, profile):
        tool = MemoryWriteTool()
        tool._run(content="First fact")
        tool._run(content="Second fact")

        content = profile.memory_file.read_text()
        assert "First fact" in content
        assert "Second fact" in content

    def test_write_with_comment(self, profile):
        tool = MemoryWriteTool()
        tool._run(content="Timezone is UTC+9", comment="User corrected from UTC+8")

        content = profile.memory_file.read_text()
        assert "Timezone is UTC+9" in content
        assert "User corrected from UTC+8" in content

    def test_rewrite_mode_replaces_entirely(self, profile):
        tool = MemoryWriteTool()
        tool._run(content="Old fact")
        tool._run(content="# Consolidated Memory\n\nOnly new fact", mode="rewrite")

        content = profile.memory_file.read_text()
        assert "Only new fact" in content
        assert "Old fact" not in content

    def test_rewrite_returns_confirmation(self, profile):
        tool = MemoryWriteTool()
        result = tool._run(content="# Fresh start", mode="rewrite")
        assert "Rewrote" in result


# ---------------------------------------------------------------------------
# Round-trip: write then read
# ---------------------------------------------------------------------------


class TestMemoryRoundTrip:
    def test_write_then_read(self):
        write = MemoryWriteTool()
        read = MemoryReadTool()

        write._run(content="Python 3.13 is the project version")
        result = read._run()
        assert "Python 3.13" in result

    def test_agent_and_global_roundtrip(self):
        write = MemoryWriteTool()
        read = MemoryReadTool()

        write._run(content="Agent-specific: prefers YAML", scope="agent")
        write._run(content="Global: proxy blocks DuckDuckGo", scope="global")

        result = read._run()
        assert "prefers YAML" in result
        assert "proxy blocks DuckDuckGo" in result

    def test_rewrite_then_read(self):
        write = MemoryWriteTool()
        read = MemoryReadTool()

        write._run(content="Old entry")
        write._run(content="# Clean Memory\n\nConsolidated entry", mode="rewrite")
        result = read._run()
        assert "Consolidated entry" in result
        assert "Old entry" not in result


# ---------------------------------------------------------------------------
# System prompt includes memory section
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# identity_update
# ---------------------------------------------------------------------------


class TestIdentityUpdate:
    def test_update_creates_identity(self, profile):
        tool = IdentityUpdateTool()
        result = tool._run(content="# User Profile\n\n- Name: Alice\n- Role: Developer")
        assert "updated" in result

        loaded = profile.load_identity()
        assert "Alice" in loaded
        assert "Developer" in loaded

    def test_update_replaces_entirely(self, profile):
        tool = IdentityUpdateTool()
        tool._run(content="# V1\nOld data")
        tool._run(content="# V2\nNew data")

        loaded = profile.load_identity()
        assert "New data" in loaded
        assert "Old data" not in loaded

    def test_no_profile_returns_message(self, monkeypatch):
        monkeypatch.setattr(ap, "_current_profile", None)
        tool = IdentityUpdateTool()
        result = tool._run(content="test")
        assert "not available" in result


# ---------------------------------------------------------------------------
# System prompt includes memory & identity section
# ---------------------------------------------------------------------------


class TestSystemPromptMemory:
    def test_system_prompt_contains_memory_and_identity(self):
        from cliver.llm.base import LLMInferenceEngine

        prompt = LLMInferenceEngine._section_interaction_guidelines()
        assert "Memory & Identity" in prompt
        assert "MemoryRead" in prompt
        assert "MemoryWrite" in prompt
        assert "Identity" in prompt
