"""App-layer path resolution tests."""

from pathlib import Path

from cliver.paths import get_config_dir


def test_conf_dir_env_override_wins(tmp_path: Path) -> None:
    override = tmp_path / "custom-conf"
    assert (
        get_config_dir(
            env={"CLIVER_CONF_DIR": str(override)},
            home=tmp_path,
            system="Darwin",
        )
        == override
    )


def test_unix_defaults_to_hidden_config_dir(tmp_path: Path) -> None:
    for system in ("Darwin", "Linux"):
        assert get_config_dir(env={}, home=tmp_path, system=system) == tmp_path / ".cliver"


def test_windows_uses_appdata(tmp_path: Path) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    assert (
        get_config_dir(
            env={"APPDATA": str(appdata)},
            home=tmp_path,
            system="Windows",
        )
        == appdata / "cliver"
    )


def test_windows_without_appdata_falls_back_to_home(tmp_path: Path) -> None:
    assert (
        get_config_dir(
            env={},
            home=tmp_path,
            system="Windows",
        )
        == tmp_path / "AppData" / "Roaming" / "cliver"
    )
