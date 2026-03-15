"""Tests for memory_read and memory_write builtin tools."""

import pytest

import cliver.agent_profile as ap
from cliver.agent_profile import AgentProfile
from cliver.tools.memory import MemoryReadTool, MemoryWriteTool


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
        result = tool._run(entry="User uses uv, not pip")
        assert "Saved to agent memory" in result

        content = profile.memory_file.read_text()
        assert "User uses uv, not pip" in content

    def test_write_global_memory(self, profile, tmp_path):
        tool = MemoryWriteTool()
        result = tool._run(entry="Shared knowledge", scope="global")
        assert "Saved to global memory" in result

        content = (tmp_path / "memory.md").read_text()
        assert "Shared knowledge" in content

    def test_default_scope_is_agent(self, profile):
        tool = MemoryWriteTool()
        tool._run(entry="Should be agent-scoped")

        assert profile.memory_file.exists()
        content = profile.memory_file.read_text()
        assert "Should be agent-scoped" in content

    def test_no_profile_returns_message(self, monkeypatch):
        monkeypatch.setattr(ap, "_current_profile", None)
        tool = MemoryWriteTool()
        result = tool._run(entry="test")
        assert "not available" in result

    def test_multiple_writes_are_additive(self, profile):
        tool = MemoryWriteTool()
        tool._run(entry="First fact")
        tool._run(entry="Second fact")

        content = profile.memory_file.read_text()
        assert "First fact" in content
        assert "Second fact" in content


# ---------------------------------------------------------------------------
# Round-trip: write then read
# ---------------------------------------------------------------------------


class TestMemoryRoundTrip:
    def test_write_then_read(self):
        write = MemoryWriteTool()
        read = MemoryReadTool()

        write._run(entry="Python 3.13 is the project version")
        result = read._run()
        assert "Python 3.13" in result

    def test_agent_and_global_roundtrip(self):
        write = MemoryWriteTool()
        read = MemoryReadTool()

        write._run(entry="Agent-specific: prefers YAML", scope="agent")
        write._run(entry="Global: proxy blocks DuckDuckGo", scope="global")

        result = read._run()
        assert "prefers YAML" in result
        assert "proxy blocks DuckDuckGo" in result


# ---------------------------------------------------------------------------
# System prompt includes memory section
# ---------------------------------------------------------------------------


class TestSystemPromptMemory:
    def test_system_prompt_contains_memory_section(self):
        from cliver.llm.base import LLMInferenceEngine

        prompt = LLMInferenceEngine._section_interaction_guidelines()
        assert "## Memory" in prompt
        assert "memory_read" in prompt
        assert "memory_write" in prompt
