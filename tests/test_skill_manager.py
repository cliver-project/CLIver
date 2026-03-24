"""Tests for the SkillManager and skill builtin tool."""

import textwrap
from pathlib import Path

import pytest

from cliver.skill_manager import (
    Skill,
    SkillManager,
    _parse_skill_md,
    validate_skill,
    validate_skill_name,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "skills"


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory with test skills."""
    skills = tmp_path / "skills"

    # Skill 1: well-formed with spec fields
    s1 = skills / "web-search"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: web-search
        description: Search the web for information.
        license: Apache-2.0
        compatibility: Requires internet access
        metadata:
          author: test-org
          version: "1.0"
        allowed-tools: web_search web_fetch
        ---

        # Web Search

        Use tavily_search to find information online.
    """)
    )

    # Skill 2: minimal
    s2 = skills / "summarizer"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: summarizer
        description: Summarize text.
        ---

        Summarize the given text concisely.
    """)
    )

    # Non-skill directory (no SKILL.md)
    nonskill = skills / "not-a-skill"
    nonskill.mkdir(parents=True)
    (nonskill / "README.md").write_text("Not a skill")

    return skills


@pytest.fixture
def manager(skills_dir, monkeypatch):
    """Create a SkillManager that uses the test skills directory."""
    # Patch get_config_dir to return a dir that doesn't have skills
    monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: skills_dir.parent / "no-global")
    # Patch cwd to point .cliver/skills to our test dir
    monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: skills_dir.parent)
    # Create the .cliver/skills structure
    local_skills = skills_dir.parent / ".cliver" / "skills"
    local_skills.mkdir(parents=True, exist_ok=True)
    # Copy test skills into .cliver/skills
    import shutil

    for child in skills_dir.iterdir():
        if child.is_dir():
            dest = local_skills / child.name
            if not dest.exists():
                shutil.copytree(child, dest)

    mgr = SkillManager()
    return mgr


# ---------------------------------------------------------------------------
# validate_skill_name
# ---------------------------------------------------------------------------


class TestValidateSkillName:
    def test_valid_names(self):
        assert validate_skill_name("pdf-processing") == []
        assert validate_skill_name("data-analysis") == []
        assert validate_skill_name("code-review") == []
        assert validate_skill_name("a") == []
        assert validate_skill_name("tool123") == []

    def test_empty_name(self):
        errors = validate_skill_name("")
        assert any("empty" in e for e in errors)

    def test_uppercase(self):
        errors = validate_skill_name("PDF-Processing")
        assert len(errors) > 0
        assert any("lowercase" in e for e in errors)

    def test_leading_hyphen(self):
        errors = validate_skill_name("-pdf")
        assert len(errors) > 0

    def test_trailing_hyphen(self):
        errors = validate_skill_name("pdf-")
        assert len(errors) > 0

    def test_consecutive_hyphens(self):
        errors = validate_skill_name("pdf--processing")
        assert len(errors) > 0
        assert any("consecutive" in e for e in errors)

    def test_spaces(self):
        errors = validate_skill_name("web search")
        assert len(errors) > 0

    def test_too_long(self):
        errors = validate_skill_name("a" * 65)
        assert any("64" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_skill
# ---------------------------------------------------------------------------


class TestValidateSkill:
    def test_valid_skill(self):
        skill = Skill(name="test-skill", description="A test skill.", body="content", base_dir=Path("test-skill"))
        result = validate_skill(skill)
        assert result.is_valid
        assert len(result.warnings) == 0

    def test_name_dir_mismatch_warns(self):
        skill = Skill(name="test-skill", description="A test.", body="", base_dir=Path("different-name"))
        result = validate_skill(skill)
        assert any("does not match directory" in w for w in result.warnings)

    def test_long_description_warns(self):
        skill = Skill(name="test", description="x" * 1025, body="", base_dir=Path("test"))
        result = validate_skill(skill)
        assert any("1024" in w for w in result.warnings)

    def test_long_compatibility_warns(self):
        skill = Skill(name="test", description="ok", body="", base_dir=Path("test"), compatibility="x" * 501)
        result = validate_skill(skill)
        assert any("500" in w for w in result.warnings)

    def test_long_body_warns(self):
        skill = Skill(name="test", description="ok", body="\n".join(["line"] * 501), base_dir=Path("test"))
        result = validate_skill(skill)
        assert any("500 lines" in w for w in result.warnings)

    def test_missing_description_errors(self):
        skill = Skill(name="test", description="", body="content", base_dir=Path("test"))
        result = validate_skill(skill)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# _parse_skill_md
# ---------------------------------------------------------------------------


class TestParseSkillMd:
    def test_parses_valid_skill(self):
        path = FIXTURES_DIR / "test-greeting" / "SKILL.md"
        skill = _parse_skill_md(path)
        assert skill is not None
        assert skill.name == "test-greeting"
        assert "Greet the user" in skill.description
        assert "# Greeting Skill" in skill.body
        assert skill.base_dir == FIXTURES_DIR / "test-greeting"

    def test_frontmatter_fields(self):
        path = FIXTURES_DIR / "test-greeting" / "SKILL.md"
        skill = _parse_skill_md(path)
        assert skill.allowed_tools == ["web_search"]
        assert skill.frontmatter["parameters"] == {"default_language": "english"}

    def test_missing_frontmatter(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("No frontmatter here")
        skill = _parse_skill_md(path)
        assert skill is None

    def test_missing_name_field(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\ndescription: no name\n---\nbody")
        skill = _parse_skill_md(path)
        assert skill is None

    def test_empty_body(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: empty\ndescription: empty body\n---\n")
        skill = _parse_skill_md(path)
        assert skill is not None
        assert skill.body == ""

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "nonexistent" / "SKILL.md"
        skill = _parse_skill_md(path)
        assert skill is None

    def test_spec_allowed_tools_string(self, tmp_path):
        """allowed-tools as space-delimited string per spec."""
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: test\ndescription: test\nallowed-tools: Bash Read Write\n---\nbody")
        skill = _parse_skill_md(path)
        assert skill.allowed_tools == ["Bash", "Read", "Write"]

    def test_allowed_tools_array(self, tmp_path):
        """allowed-tools as YAML array (also accepted)."""
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: test\ndescription: test\nallowed-tools:\n  - tool1\n  - tool2\n---\nbody")
        skill = _parse_skill_md(path)
        assert skill.allowed_tools == ["tool1", "tool2"]

    def test_optional_fields_parsed(self, tmp_path):
        """license, compatibility, metadata are parsed."""
        path = tmp_path / "SKILL.md"
        path.write_text(
            textwrap.dedent("""\
            ---
            name: full-skill
            description: A fully specified skill.
            license: MIT
            compatibility: Requires Python 3.10+
            metadata:
              author: test
              version: "2.0"
            ---
            body
        """)
        )
        skill = _parse_skill_md(path)
        assert skill.license == "MIT"
        assert skill.compatibility == "Requires Python 3.10+"
        assert skill.metadata == {"author": "test", "version": "2.0"}

    def test_tolerant_of_uppercase_names(self, tmp_path):
        """Claude Code skills use uppercase names like 'Fetch Payloads' — should load with warning."""
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: Fetch Payloads\ndescription: Fetches payloads.\n---\nbody")
        skill = _parse_skill_md(path)
        assert skill is not None
        assert skill.name == "Fetch Payloads"

    def test_bom_handling(self, tmp_path):
        """Files with BOM should parse correctly."""
        path = tmp_path / "SKILL.md"
        path.write_text("\ufeff---\nname: bom-test\ndescription: BOM test.\n---\nbody")
        skill = _parse_skill_md(path)
        assert skill is not None
        assert skill.name == "bom-test"

    def test_crlf_line_endings(self, tmp_path):
        """Files with CRLF should parse correctly."""
        path = tmp_path / "SKILL.md"
        path.write_bytes(b"---\r\nname: crlf\r\ndescription: CRLF test.\r\n---\r\nbody")
        skill = _parse_skill_md(path)
        assert skill is not None
        assert skill.name == "crlf"


# ---------------------------------------------------------------------------
# SkillManager discovery
# ---------------------------------------------------------------------------


class TestSkillManagerDiscovery:
    def test_discovers_skills(self, manager):
        names = manager.get_skill_names()
        assert "web-search" in names
        assert "summarizer" in names

    def test_ignores_non_skill_dirs(self, manager):
        names = manager.get_skill_names()
        assert "not-a-skill" not in names

    def test_list_skills_returns_skill_objects(self, manager):
        skills = manager.list_skills()
        assert all(isinstance(s, Skill) for s in skills)
        assert len(skills) == 2

    def test_get_skill_by_name(self, manager):
        skill = manager.get_skill("web-search")
        assert skill is not None
        assert skill.name == "web-search"
        assert "tavily_search" in skill.body

    def test_get_nonexistent_skill(self, manager):
        skill = manager.get_skill("nonexistent")
        assert skill is None

    def test_spec_fields_preserved(self, manager):
        skill = manager.get_skill("web-search")
        assert skill.license == "Apache-2.0"
        assert skill.compatibility == "Requires internet access"
        assert skill.metadata == {"author": "test-org", "version": "1.0"}
        assert skill.allowed_tools == ["web_search", "web_fetch"]


# ---------------------------------------------------------------------------
# Multi-source discovery
# ---------------------------------------------------------------------------


class TestMultiSourceDiscovery:
    def test_agent_skills_dir(self, tmp_path, monkeypatch):
        """Skills from .agent/skills/ are discovered."""
        agent_dir = tmp_path / "project" / ".agent" / "skills" / "my-tool"
        agent_dir.mkdir(parents=True)
        (agent_dir / "SKILL.md").write_text("---\nname: my-tool\ndescription: Agent skill.\n---\nbody")

        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "no-global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "project")

        mgr = SkillManager()
        skill = mgr.get_skill("my-tool")
        assert skill is not None
        assert "project (.agent)" in skill.source

    def test_global_agents_dir(self, tmp_path, monkeypatch):
        """Skills from ~/.agents/skills/ are discovered."""
        agents_dir = tmp_path / "home" / ".agents" / "skills" / "global-tool"
        agents_dir.mkdir(parents=True)
        (agents_dir / "SKILL.md").write_text("---\nname: global-tool\ndescription: Global agent skill.\n---\nbody")

        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "no-config")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "no-project")
        monkeypatch.setattr("cliver.skill_manager.Path.home", lambda: tmp_path / "home")

        mgr = SkillManager()
        skill = mgr.get_skill("global-tool")
        assert skill is not None
        assert "global (agents)" in skill.source

    def test_claude_compat_dir(self, tmp_path, monkeypatch):
        """Skills from .claude/skills/ are discovered for cross-agent compat."""
        claude_dir = tmp_path / "project" / ".claude" / "skills" / "claude-skill"
        claude_dir.mkdir(parents=True)
        (claude_dir / "SKILL.md").write_text("---\nname: claude-skill\ndescription: From Claude.\n---\nbody")

        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "no-global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "project")

        mgr = SkillManager()
        skill = mgr.get_skill("claude-skill")
        assert skill is not None
        assert "claude" in skill.source

    def test_qwen_compat_dir(self, tmp_path, monkeypatch):
        """Skills from .qwen/skills/ are discovered for cross-agent compat."""
        qwen_dir = tmp_path / "project" / ".qwen" / "skills" / "qwen-skill"
        qwen_dir.mkdir(parents=True)
        (qwen_dir / "SKILL.md").write_text("---\nname: qwen-skill\ndescription: From Qwen.\n---\nbody")

        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "no-global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "project")

        mgr = SkillManager()
        skill = mgr.get_skill("qwen-skill")
        assert skill is not None
        assert "qwen" in skill.source

    def test_project_overrides_global(self, tmp_path, monkeypatch):
        """Project-local skills override global skills with the same name."""
        global_dir = tmp_path / "global" / "skills" / "my-skill"
        global_dir.mkdir(parents=True)
        (global_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: global version\n---\nGlobal body")

        local_dir = tmp_path / "project" / ".cliver" / "skills" / "my-skill"
        local_dir.mkdir(parents=True)
        (local_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: local version\n---\nLocal body")

        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "project")

        mgr = SkillManager()
        skill = mgr.get_skill("my-skill")
        assert skill is not None
        assert skill.description == "local version"
        assert "Local body" in skill.body


# ---------------------------------------------------------------------------
# SkillManager.activate_skill
# ---------------------------------------------------------------------------


class TestActivateSkill:
    def test_activate_returns_body_with_base_dir(self, manager):
        result = manager.activate_skill("web-search")
        assert "Base directory for this skill:" in result
        assert "# Web Search" in result
        assert "tavily_search" in result

    def test_activate_nonexistent_lists_available(self, manager):
        result = manager.activate_skill("nonexistent")
        assert "not found" in result
        assert "web-search" in result
        assert "summarizer" in result


# ---------------------------------------------------------------------------
# SkillManager.format_skill_list
# ---------------------------------------------------------------------------


class TestFormatSkillList:
    def test_lists_all_skills(self, manager):
        output = manager.format_skill_list()
        assert "web-search" in output
        assert "summarizer" in output
        assert "Search the web" in output

    def test_includes_count(self, manager):
        output = manager.format_skill_list()
        assert "(2)" in output

    def test_includes_activation_hint(self, manager):
        output = manager.format_skill_list()
        assert "skill(" in output

    def test_empty_skills(self, monkeypatch):
        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: Path("/nonexistent"))
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: Path("/nonexistent"))
        mgr = SkillManager()
        assert mgr.format_skill_list() == "No skills available."

    def test_long_description_truncated(self, tmp_path, monkeypatch):
        """Long descriptions are truncated in the list view."""
        skill_dir = tmp_path / "project" / ".cliver" / "skills" / "verbose"
        skill_dir.mkdir(parents=True)
        long_desc = "A" * 200
        (skill_dir / "SKILL.md").write_text(f"---\nname: verbose\ndescription: {long_desc}\n---\nbody")

        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "no-global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "project")

        mgr = SkillManager()
        output = mgr.format_skill_list()
        # Should be truncated
        assert "..." in output


# ---------------------------------------------------------------------------
# SkillManager.reload
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_clears_cache(self, manager):
        assert len(manager.list_skills()) == 2
        manager.reload()
        # After reload, skills are re-discovered (same result since dirs unchanged)
        assert len(manager.list_skills()) == 2


# ---------------------------------------------------------------------------
# Skill builtin tool
# ---------------------------------------------------------------------------


class TestSkillTool:
    def _patch_manager(self, manager, monkeypatch):
        """Patch the module-level _skill_manager used by the skill tool."""
        import sys

        # Get the actual module (not the re-exported tool instance)
        mod = sys.modules.get("cliver.tools.skill")
        if mod is None:
            import cliver.tools.skill as mod  # noqa: F811
        monkeypatch.setattr(mod, "_skill_manager", manager)

    def test_tool_list(self, manager, monkeypatch):
        """The skill tool should list available skills when called with 'list'."""
        self._patch_manager(manager, monkeypatch)
        from cliver.tools.skill import SkillTool

        tool = SkillTool()
        result = tool._run(skill_name="list")
        assert "web-search" in result
        assert "summarizer" in result

    def test_tool_activate(self, manager, monkeypatch):
        """The skill tool should return skill body when called with a name."""
        self._patch_manager(manager, monkeypatch)
        from cliver.tools.skill import SkillTool

        tool = SkillTool()
        result = tool._run(skill_name="web-search")
        assert "# Web Search" in result
        assert "Base directory" in result

    def test_tool_not_found(self, manager, monkeypatch):
        self._patch_manager(manager, monkeypatch)
        from cliver.tools.skill import SkillTool

        tool = SkillTool()
        result = tool._run(skill_name="nonexistent")
        assert "not found" in result
