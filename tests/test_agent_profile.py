"""Tests for AgentProfile — instance-scoped resource management."""

import pytest

from cliver.agent_profile import MAX_MEMORY_CHARS, AgentProfile


@pytest.fixture
def profile(tmp_path):
    """Create an AgentProfile with a temp config directory."""
    return AgentProfile("TestBot", config_dir=tmp_path)


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------


class TestDirectoryStructure:
    def test_paths_use_agent_name(self, profile, tmp_path):
        assert profile.agent_dir == tmp_path / "agents" / "TestBot"
        assert profile.memory_file == tmp_path / "agents" / "TestBot" / "memory.md"
        assert profile.identity_file == tmp_path / "agents" / "TestBot" / "identity.md"
        assert profile.tasks_dir == tmp_path / "agents" / "TestBot" / "tasks"
        assert profile.sessions_dir == tmp_path / "agents" / "TestBot" / "sessions"

    def test_global_memory_path(self, profile, tmp_path):
        assert profile.global_memory_file == tmp_path / "memory.md"

    def test_ensure_dirs_creates_agent_dir(self, profile):
        assert not profile.agent_dir.exists()
        profile.ensure_dirs()
        assert profile.agent_dir.exists()


# ---------------------------------------------------------------------------
# Memory — loading
# ---------------------------------------------------------------------------


class TestLoadMemory:
    def test_empty_when_no_files(self, profile):
        assert profile.load_memory() == ""

    def test_loads_global_memory_only(self, profile, tmp_path):
        (tmp_path / "memory.md").write_text("Global note")
        result = profile.load_memory()
        assert "Global Memory" in result
        assert "Global note" in result

    def test_loads_agent_memory_only(self, profile):
        profile.ensure_dirs()
        profile.memory_file.write_text("Agent note")
        result = profile.load_memory()
        assert "Agent Memory: TestBot" in result
        assert "Agent note" in result

    def test_loads_both_memories(self, profile, tmp_path):
        (tmp_path / "memory.md").write_text("Global info")
        profile.ensure_dirs()
        profile.memory_file.write_text("Agent info")

        result = profile.load_memory()
        assert "Global Memory" in result
        assert "Global info" in result
        assert "Agent Memory: TestBot" in result
        assert "Agent info" in result

    def test_global_comes_before_agent(self, profile, tmp_path):
        (tmp_path / "memory.md").write_text("Global")
        profile.ensure_dirs()
        profile.memory_file.write_text("Agent")

        result = profile.load_memory()
        global_pos = result.index("Global Memory")
        agent_pos = result.index("Agent Memory")
        assert global_pos < agent_pos

    def test_truncates_long_memory(self, profile, tmp_path):
        # Create memory that exceeds MAX_MEMORY_CHARS
        long_content = "x" * (MAX_MEMORY_CHARS + 1000)
        (tmp_path / "memory.md").write_text(long_content)

        result = profile.load_memory()
        assert len(result) <= MAX_MEMORY_CHARS + 100  # some overhead for headers
        assert result.startswith("...(truncated)")


# ---------------------------------------------------------------------------
# Memory — appending
# ---------------------------------------------------------------------------


class TestAppendMemory:
    def test_append_creates_agent_memory_file(self, profile):
        assert not profile.memory_file.exists()
        profile.append_memory("User prefers dark mode")
        assert profile.memory_file.exists()

        content = profile.memory_file.read_text()
        assert "Memory: TestBot" in content
        assert "User prefers dark mode" in content

    def test_append_adds_timestamp(self, profile):
        profile.append_memory("Some fact")
        content = profile.memory_file.read_text()
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", content)

    def test_append_is_additive(self, profile):
        profile.append_memory("First entry")
        profile.append_memory("Second entry")

        content = profile.memory_file.read_text()
        assert "First entry" in content
        assert "Second entry" in content

    def test_append_global_memory(self, profile, tmp_path):
        profile.append_memory("Shared knowledge", scope="global")
        assert (tmp_path / "memory.md").exists()

        content = (tmp_path / "memory.md").read_text()
        assert "Global Memory" in content
        assert "Shared knowledge" in content

    def test_append_global_does_not_create_agent_dir(self, profile):
        profile.append_memory("Global only", scope="global")
        assert not profile.agent_dir.exists()

    def test_append_with_comment(self, profile):
        profile.append_memory("Timezone is UTC+9", comment="Corrected from UTC+8")
        content = profile.memory_file.read_text()
        assert "Timezone is UTC+9" in content
        assert "Corrected from UTC+8" in content
        assert "UTC" in content  # timestamp still present

    def test_save_memory_replaces_entirely(self, profile):
        profile.append_memory("Old entry")
        profile.save_memory("# Fresh Memory\n\nOnly this remains")
        content = profile.memory_file.read_text()
        assert "Only this remains" in content
        assert "Old entry" not in content


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_empty_identity_when_no_file(self, profile):
        assert profile.load_identity() == ""

    def test_load_identity(self, profile):
        profile.ensure_dirs()
        profile.identity_file.write_text("# User Profile\n\n- Name: Alice\n- Location: Tokyo\n")
        identity = profile.load_identity()
        assert "Alice" in identity
        assert "Tokyo" in identity

    def test_save_and_load_identity(self, profile):
        content = "# User Profile\n\n- Prefers concise responses\n- Uses Python 3.13\n"
        profile.save_identity(content)
        loaded = profile.load_identity()
        assert "concise responses" in loaded
        assert "Python 3.13" in loaded

    def test_save_replaces_entirely(self, profile):
        profile.save_identity("# Version 1\nOld info")
        profile.save_identity("# Version 2\nNew info")
        loaded = profile.load_identity()
        assert "New info" in loaded
        assert "Old info" not in loaded

    def test_save_creates_agent_dir(self, profile):
        assert not profile.agent_dir.exists()
        profile.save_identity("# Test")
        assert profile.agent_dir.exists()

    def test_identity_file_is_markdown(self, profile):
        assert profile.identity_file.name == "identity.md"


# ---------------------------------------------------------------------------
# Multiple agents — isolation
# ---------------------------------------------------------------------------


class TestAgentIsolation:
    def test_different_agents_have_separate_memory(self, tmp_path):
        bot_a = AgentProfile("BotA", config_dir=tmp_path)
        bot_b = AgentProfile("BotB", config_dir=tmp_path)

        bot_a.append_memory("I am Bot A")
        bot_b.append_memory("I am Bot B")

        mem_a = bot_a.load_memory()
        mem_b = bot_b.load_memory()

        assert "I am Bot A" in mem_a
        assert "I am Bot B" not in mem_a
        assert "I am Bot B" in mem_b
        assert "I am Bot A" not in mem_b

    def test_agents_share_global_memory(self, tmp_path):
        bot_a = AgentProfile("BotA", config_dir=tmp_path)
        bot_b = AgentProfile("BotB", config_dir=tmp_path)

        bot_a.append_memory("Shared fact", scope="global")

        mem_a = bot_a.load_memory()
        mem_b = bot_b.load_memory()

        assert "Shared fact" in mem_a
        assert "Shared fact" in mem_b

    def test_agents_have_separate_identity(self, tmp_path):
        bot_a = AgentProfile("BotA", config_dir=tmp_path)
        bot_b = AgentProfile("BotB", config_dir=tmp_path)

        bot_a.save_identity("# BotA\nPersona: helpful")
        bot_b.save_identity("# BotB\nPersona: concise")

        assert "helpful" in bot_a.load_identity()
        assert "concise" in bot_b.load_identity()
        assert "helpful" not in bot_b.load_identity()


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------


class TestRename:
    def test_rename_moves_data(self, tmp_path):
        old = AgentProfile("OldName", config_dir=tmp_path)
        old.append_memory("Remember this")
        old.save_identity("# Profile\nPersona: original")

        new = old.rename("NewName")

        assert new.agent_name == "NewName"
        assert new.agent_dir == tmp_path / "agents" / "NewName"
        assert not old.agent_dir.exists()
        assert new.agent_dir.exists()

        # Data is preserved
        mem = new.load_memory()
        assert "Remember this" in mem
        assert "original" in new.load_identity()

    def test_rename_target_exists_raises(self, tmp_path):
        a = AgentProfile("AgentA", config_dir=tmp_path)
        b = AgentProfile("AgentB", config_dir=tmp_path)
        a.ensure_dirs()
        b.ensure_dirs()

        with pytest.raises(FileExistsError):
            a.rename("AgentB")

    def test_rename_no_existing_data_returns_fresh(self, tmp_path):
        old = AgentProfile("Ghost", config_dir=tmp_path)
        # Don't create any data
        new = old.rename("Fresh")
        assert new.agent_name == "Fresh"
        assert new.load_memory() == ""
        assert new.load_identity() == ""
