"""Tests for CliverProfile — instance-scoped resource management."""

import pytest

from cliver.agent_profile import MAX_MEMORY_CHARS, CliverProfile, _parse_frontmatter


@pytest.fixture
def profile(tmp_path):
    """Create a CliverProfile with a temp config directory."""
    return CliverProfile(config_dir=tmp_path)


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------


class TestDirectoryStructure:
    def test_paths_are_flat(self, profile, tmp_path):
        assert profile.memory_file == tmp_path / "memory.md"
        assert profile.identity_file == tmp_path / "identity.md"
        assert profile.tasks_dir == tmp_path / "tasks"
        assert profile.sessions_dir == tmp_path / "sessions"
        assert profile.gateway_db == tmp_path / "gateway.db"

    def test_ensure_dirs_creates_config_dir(self, profile, tmp_path):
        profile.ensure_dirs()
        assert tmp_path.exists()


# ---------------------------------------------------------------------------
# Memory — loading
# ---------------------------------------------------------------------------


class TestLoadMemory:
    def test_empty_when_no_files(self, profile):
        assert profile.load_memory() == ""

    def test_loads_memory(self, profile, tmp_path):
        (tmp_path / "memory.md").write_text("User prefers dark mode")
        result = profile.load_memory()
        assert "User prefers dark mode" in result

    def test_truncates_long_memory(self, profile, tmp_path):
        long_content = "x" * (MAX_MEMORY_CHARS + 1000)
        (tmp_path / "memory.md").write_text(long_content)

        result = profile.load_memory()
        assert len(result) <= MAX_MEMORY_CHARS + 100
        assert result.startswith("...(truncated)")


# ---------------------------------------------------------------------------
# Memory — appending
# ---------------------------------------------------------------------------


class TestAppendMemory:
    def test_append_creates_memory_file(self, profile):
        assert not profile.memory_file.exists()
        profile.append_memory("User prefers dark mode")
        assert profile.memory_file.exists()

        content = profile.memory_file.read_text()
        assert "Memory" in content
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

    def test_append_with_comment(self, profile):
        profile.append_memory("Timezone is UTC+9", comment="Corrected from UTC+8")
        content = profile.memory_file.read_text()
        assert "Timezone is UTC+9" in content
        assert "Corrected from UTC+8" in content

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

    def test_load_identity(self, profile, tmp_path):
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

    def test_identity_file_is_markdown(self, profile):
        assert profile.identity_file.name == "identity.md"


# ---------------------------------------------------------------------------
# Profile — YAML frontmatter
# ---------------------------------------------------------------------------


class TestProfile:
    def test_empty_profile_when_no_file(self, profile):
        assert profile.load_profile() == {}

    def test_load_profile_from_frontmatter(self, profile):
        profile.save_identity("---\nname: Alice\nrole: Engineer\n---\n\n## Notes\nSome text")
        data = profile.load_profile()
        assert data["name"] == "Alice"
        assert data["role"] == "Engineer"

    def test_set_profile_field(self, profile):
        profile.save_identity("---\nname: CLIver\n---\n")
        profile.set_profile_field("role", "Engineer")
        data = profile.load_profile()
        assert data["name"] == "CLIver"
        assert data["role"] == "Engineer"

    def test_set_nested_field(self, profile):
        profile.save_identity("---\nname: CLIver\n---\n")
        profile.set_profile_field("preferences.language", "zh-CN")
        data = profile.load_profile()
        assert data["preferences"]["language"] == "zh-CN"

    def test_set_preserves_body(self, profile):
        profile.save_identity("---\nname: CLIver\n---\n\n## My Notes\nKeep this text")
        profile.set_profile_field("role", "Dev")
        content = profile.load_identity()
        assert "My Notes" in content
        assert "Keep this text" in content
        assert "role: Dev" in content

    def test_profile_name_property(self, profile):
        assert profile.profile_name == "CLIver"  # default
        profile.save_identity("---\nname: Alice\n---\n")
        assert profile.profile_name == "Alice"

    def test_set_creates_file_if_missing(self, profile):
        assert not profile.identity_file.exists()
        profile.set_profile_field("name", "Bob")
        assert profile.identity_file.exists()
        assert profile.load_profile()["name"] == "Bob"


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_parse_empty(self):
        fm, body = _parse_frontmatter("")
        assert fm == {}
        assert body == ""

    def test_parse_no_frontmatter(self):
        fm, body = _parse_frontmatter("# Just markdown")
        assert fm == {}
        assert body == "# Just markdown"

    def test_parse_with_frontmatter(self):
        content = "---\nname: Test\nrole: Dev\n---\n\n## Body"
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "Test"
        assert fm["role"] == "Dev"
        assert "## Body" in body

    def test_parse_frontmatter_only(self):
        content = "---\nname: Test\n---"
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "Test"
        assert body == ""
