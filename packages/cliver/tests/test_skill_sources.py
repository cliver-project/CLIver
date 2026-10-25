"""The CLI's eight skill sources and their priority order."""

from pathlib import Path

from cliver.skills import (
    BUILTIN_SKILLS_DIR,
    create_skill_manager,
    default_skill_sources,
    source_for_skill,
)
from cliver_core.skill import SkillManager


def write_skill(root: Path, name: str, description: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody of {name}.\n",
        encoding="utf-8",
    )
    return skill_dir


def test_eight_sources_in_priority_order(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    cwd = tmp_path / "project"
    home = tmp_path / "home"

    sources = default_skill_sources(config_dir=config_dir, cwd=cwd, home=home)

    assert [(source.label, source.path) for source in sources] == [
        ("builtin", BUILTIN_SKILLS_DIR),
        ("global (cliver)", config_dir / "skills"),
        ("global (agents)", home / ".agents" / "skills"),
        ("project (.cliver)", cwd / ".cliver" / "skills"),
        ("project (.agent)", cwd / ".agent" / "skills"),
        ("project (.claude compat)", cwd / ".claude" / "skills"),
        ("project (.gemini compat)", cwd / ".gemini" / "skills"),
        ("project (.qwen compat)", cwd / ".qwen" / "skills"),
    ]


def test_manager_loads_every_source_and_overrides_by_priority(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    cwd = tmp_path / "project"
    home = tmp_path / "home"

    write_skill(config_dir / "skills", "shared", "from global cliver")
    write_skill(home / ".agents" / "skills", "agents-only", "from global agents")
    write_skill(cwd / ".cliver" / "skills", "shared", "from project cliver")
    write_skill(cwd / ".qwen" / "skills", "qwen-only", "from qwen")

    manager = create_skill_manager(config_dir=config_dir, cwd=cwd, home=home)

    assert manager.get_skill("shared").description == "from project cliver"
    assert manager.get_skill("agents-only").description == "from global agents"
    assert manager.get_skill("qwen-only").description == "from qwen"


def test_qwen_overrides_other_project_sources(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    cwd = tmp_path / "project"
    home = tmp_path / "home"

    write_skill(cwd / ".gemini" / "skills", "shared", "from gemini")
    write_skill(cwd / ".qwen" / "skills", "shared", "from qwen")

    manager = create_skill_manager(config_dir=config_dir, cwd=cwd, home=home)

    assert manager.get_skill("shared").description == "from qwen"


def test_project_skill_overrides_builtin(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    write_skill(cwd / ".cliver" / "skills", "brainstorm", "project brainstorm")

    manager = create_skill_manager(
        config_dir=tmp_path / "conf",
        cwd=cwd,
        home=tmp_path / "home",
    )

    assert manager.get_skill("brainstorm").description == "project brainstorm"


def test_config_dir_defaults_to_cliver_conf_dir(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "conf"
    monkeypatch.setenv("CLIVER_CONF_DIR", str(override))

    sources = default_skill_sources(cwd=tmp_path / "project", home=tmp_path / "home")

    assert sources[1].path == override / "skills"


def test_builtin_skills_are_discovered(tmp_path: Path) -> None:
    manager = create_skill_manager(
        config_dir=tmp_path / "conf",
        cwd=tmp_path / "project",
        home=tmp_path / "home",
    )

    names = {skill.name for skill in manager.list_skills()}
    assert {"brainstorm", "execute-plan", "wbs-planner", "write-plan"} <= names


def test_missing_sources_are_ignored(tmp_path: Path) -> None:
    sources = default_skill_sources(
        config_dir=tmp_path / "conf",
        cwd=tmp_path / "project",
        home=tmp_path / "home",
    )
    manager = create_skill_manager(
        config_dir=tmp_path / "conf",
        cwd=tmp_path / "project",
        home=tmp_path / "home",
    )

    assert manager.list_skills()
    assert {source_for_skill(skill, sources).label for skill in manager.list_skills()} == {"builtin"}


def test_source_labels_follow_the_winning_source(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    cwd = tmp_path / "project"
    home = tmp_path / "home"

    write_skill(config_dir / "skills", "shared", "from global cliver")
    write_skill(cwd / ".claude" / "skills", "shared", "from claude")

    sources = default_skill_sources(config_dir=config_dir, cwd=cwd, home=home)
    manager = create_skill_manager(config_dir=config_dir, cwd=cwd, home=home)
    shared = manager.get_skill("shared")

    assert shared is not None
    assert source_for_skill(shared, sources).label == "project (.claude compat)"


def test_source_for_skill_returns_none_outside_known_sources(tmp_path: Path) -> None:
    sources = default_skill_sources(
        config_dir=tmp_path / "conf",
        cwd=tmp_path / "project",
        home=tmp_path / "home",
    )
    write_skill(tmp_path / "elsewhere", "stray", "not a known source")
    stray = SkillManager(sources=[tmp_path / "elsewhere"]).get_skill("stray")

    assert stray is not None
    assert source_for_skill(stray, sources) is None
