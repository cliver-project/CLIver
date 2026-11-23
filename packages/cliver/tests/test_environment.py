"""The CLI loads ``{config_dir}/.env`` with no external dotenv package."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from cliver.app import load_environment


def _write_env(tmp_path, text: str) -> None:
    (tmp_path / ".env").write_text(text, encoding="utf-8")


def test_load_environment_needs_no_dotenv_package(tmp_path):
    _write_env(tmp_path, "CLIVER_ENV_SUBPROCESS=from-file\n")
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import os
        import sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "dotenv":
                    raise ImportError("dotenv is not installed")
                return None

        sys.meta_path.insert(0, Blocker())
        os.environ.pop("CLIVER_ENV_SUBPROCESS", None)

        from cliver.app import load_environment
        load_environment({str(tmp_path)!r})
        assert os.environ["CLIVER_ENV_SUBPROCESS"] == "from-file"
        print("OK")
        """
    )

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_load_environment_reads_pairs_without_overriding_the_process(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIVER_ENV_PRESENT", "process")
    monkeypatch.delenv("CLIVER_ENV_NEW", raising=False)
    _write_env(tmp_path, "CLIVER_ENV_NEW=from-file\nCLIVER_ENV_PRESENT=from-file\n")

    load_environment(tmp_path)

    assert os.environ["CLIVER_ENV_NEW"] == "from-file"
    assert os.environ["CLIVER_ENV_PRESENT"] == "process"


def test_load_environment_ignores_comments_blank_lines_and_malformed_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIVER_ENV_AFTER", raising=False)
    _write_env(
        tmp_path,
        "# a comment\n\n   \n  # an indented comment\nnot a pair\n=no-key\nCLIVER_ENV_AFTER=ok\n",
    )

    load_environment(tmp_path)

    assert os.environ["CLIVER_ENV_AFTER"] == "ok"


def test_load_environment_supports_export_quotes_and_inline_comments(tmp_path, monkeypatch):
    keys = [
        "CLIVER_ENV_EXPORTED",
        "CLIVER_ENV_DOUBLE",
        "CLIVER_ENV_SINGLE",
        "CLIVER_ENV_INLINE",
        "CLIVER_ENV_HASH",
        "CLIVER_ENV_QUOTED_COMMENT",
        "CLIVER_ENV_EQUALS",
        "CLIVER_ENV_SPACED",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    _write_env(
        tmp_path,
        "export CLIVER_ENV_EXPORTED=exported\n"
        'CLIVER_ENV_DOUBLE="two words"\n'
        "CLIVER_ENV_SINGLE='single words'\n"
        "CLIVER_ENV_INLINE=value # note\n"
        'CLIVER_ENV_HASH="value # kept"\n'
        'CLIVER_ENV_QUOTED_COMMENT="two words" # note\n'
        "CLIVER_ENV_EQUALS=a=b\n"
        "CLIVER_ENV_SPACED = spaced \n",
    )

    load_environment(tmp_path)

    assert os.environ["CLIVER_ENV_EXPORTED"] == "exported"
    assert os.environ["CLIVER_ENV_DOUBLE"] == "two words"
    assert os.environ["CLIVER_ENV_SINGLE"] == "single words"
    assert os.environ["CLIVER_ENV_INLINE"] == "value"
    assert os.environ["CLIVER_ENV_HASH"] == "value # kept"
    assert os.environ["CLIVER_ENV_QUOTED_COMMENT"] == "two words"
    assert os.environ["CLIVER_ENV_EQUALS"] == "a=b"
    assert os.environ["CLIVER_ENV_SPACED"] == "spaced"


def test_load_environment_keeps_an_empty_value(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIVER_ENV_EMPTY", raising=False)
    _write_env(tmp_path, "CLIVER_ENV_EMPTY=\n")

    load_environment(tmp_path)

    assert os.environ["CLIVER_ENV_EMPTY"] == ""


def test_load_environment_missing_file_is_a_noop(tmp_path):
    load_environment(tmp_path)
