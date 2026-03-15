"""Tests for the SkillManager and skill builtin tool."""

import textwrap
from pathlib import Path

import pytest

from cliver.skill_manager import Skill, SkillManager, _parse_skill_md


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "skills"


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory with test skills."""
    skills = tmp_path / "skills"

    # Skill 1: well-formed
    s1 = skills / "web-search"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: web-search
        description: Search the web for information.
        allowedTools:
          - tavily_search
        parameters:
          max_results: 5
        ---

        # Web Search

        Use tavily_search to find information online.
    """))

    # Skill 2: minimal
    s2 = skills / "summarizer"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: summarizer
        description: Summarize text.
        ---

        Summarize the given text concisely.
    """))

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
    # Symlink test skills into .cliver/skills
    import shutil
    for child in skills_dir.iterdir():
        if child.is_dir():
            dest = local_skills / child.name
            if not dest.exists():
                shutil.copytree(child, dest)

    mgr = SkillManager()
    return mgr


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
        assert skill.frontmatter["allowedTools"] == ["web_search"]
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

    def test_empty_skills(self, monkeypatch):
        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: Path("/nonexistent"))
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: Path("/nonexistent"))
        mgr = SkillManager()
        assert mgr.format_skill_list() == "No skills available."


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
# Project-local overrides global
# ---------------------------------------------------------------------------


class TestPriorityOverride:
    def test_local_overrides_global(self, tmp_path, monkeypatch):
        """Project-local skill should override global skill with same name."""
        # Global skill
        global_dir = tmp_path / "global" / "skills" / "my-skill"
        global_dir.mkdir(parents=True)
        (global_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: global version\n---\nGlobal body"
        )

        # Local skill (same name, different content)
        local_dir = tmp_path / "project" / ".cliver" / "skills" / "my-skill"
        local_dir.mkdir(parents=True)
        (local_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: local version\n---\nLocal body"
        )

        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "project")

        mgr = SkillManager()
        skill = mgr.get_skill("my-skill")
        assert skill is not None
        assert skill.description == "local version"
        assert "Local body" in skill.body


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
