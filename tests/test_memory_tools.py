"""Tests for memory_read and memory_write builtin tools."""

import pytest

import cliver.agent_profile as ap
from cliver.agent_profile import CliverProfile
from cliver.tools.memory import MemoryReadTool, MemoryWriteTool


@pytest.fixture
def profile(tmp_path):
    """Create a CliverProfile with a temp config directory."""
    return CliverProfile(config_dir=tmp_path)


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

    def test_no_profile_returns_message(self, monkeypatch):
        monkeypatch.setattr(ap, "_current_profile", None)
        tool = MemoryReadTool()
        result = tool._run()
        assert "not available" in result


# ---------------------------------------------------------------------------
# memory_write
# ---------------------------------------------------------------------------


class TestMemoryWrite:
    def test_write_memory(self, profile):
        tool = MemoryWriteTool()
        result = tool._run(content="User uses uv, not pip")
        assert "Saved to memory" in result

        content = profile.memory_file.read_text()
        assert "User uses uv, not pip" in content

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

    def test_rewrite_then_read(self):
        write = MemoryWriteTool()
        read = MemoryReadTool()

        write._run(content="Old entry")
        write._run(content="# Clean Memory\n\nConsolidated entry", mode="rewrite")
        result = read._run()
        assert "Consolidated entry" in result
        assert "Old entry" not in result


class TestSystemPromptMemory:
    def test_system_prompt_contains_memory(self):
        from cliver.llm.base import LLMInferenceEngine

        prompt = LLMInferenceEngine._section_interaction_guidelines()
        assert "MemoryRead" in prompt
        assert "MemoryWrite" in prompt
