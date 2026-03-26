#!/usr/bin/env python3
"""
Tests for the /session option subgroup.
Tests the set/reset subcommands and default display behavior.
"""

from click.testing import CliRunner

from cliver.cli import Cliver


def _make_session_defaults(cliver_instance):
    """Create session defaults matching the reset command."""
    from cliver.config import ModelOptions

    default_options = ModelOptions()
    return {
        "model": cliver_instance.config_manager.get_llm_model(),
        "temperature": default_options.temperature,
        "max_tokens": default_options.max_tokens,
        "top_p": default_options.top_p,
        "frequency_penalty": default_options.frequency_penalty,
        "options": {},
        "stream": False,
        "save_media": False,
        "media_dir": None,
    }


def _setup(runner, load_cliver):
    """Add a test model and create a Cliver instance with session defaults."""
    runner.invoke(
        load_cliver,
        [
            "model",
            "add",
            "--name",
            "test-model",
            "--provider",
            "ollama",
            "--url",
            "http://localhost:11434",
            "--name-in-provider",
            "llama3.2:latest",
        ],
    )
    cliver_instance = Cliver()
    cliver_instance.init_session(load_cliver, _make_session_defaults(cliver_instance))
    return cliver_instance


def test_default_displays_options(load_cliver, init_config, simple_llm_model):
    """Running /session option with no subcommand displays current options."""
    runner = CliRunner()
    cliver_instance = _setup(runner, load_cliver)

    result = runner.invoke(load_cliver, ["session", "option"], obj=cliver_instance)
    assert result.exit_code == 0
    assert "Session options:" in result.output


def test_set_options(load_cliver, init_config, simple_llm_model):
    """/session option set updates options."""
    runner = CliRunner()
    cliver_instance = _setup(runner, load_cliver)

    result = runner.invoke(
        load_cliver,
        [
            "session", "option",
            "set",
            "--model",
            "test-model",
            "--temperature",
            "0.8",
            "--stream",
            "--option",
            "presence_penalty=0.5",
        ],
        obj=cliver_instance,
    )
    assert result.exit_code == 0

    # Verify by displaying
    result = runner.invoke(load_cliver, ["session", "option"], obj=cliver_instance)
    assert result.exit_code == 0
    assert "test-model" in result.output


def test_reset_subcommand(load_cliver, init_config, simple_llm_model):
    """/session option reset resets all options to defaults."""
    runner = CliRunner()
    cliver_instance = _setup(runner, load_cliver)

    # Set some values first
    runner.invoke(
        load_cliver,
        ["session", "option", "set", "--model", "test-model", "--temperature", "0.8", "--stream"],
        obj=cliver_instance,
    )

    # Reset
    result = runner.invoke(load_cliver, ["session", "option", "reset"], obj=cliver_instance)
    assert result.exit_code == 0
    assert "reset to defaults" in result.output


def test_individual_set_options(load_cliver, init_config, simple_llm_model):
    """Each option can be set individually."""
    runner = CliRunner()
    cliver_instance = _setup(runner, load_cliver)

    tests = [
        (["--model", "test-model"], "Set model to 'test-model'"),
        (["--temperature", "0.8"], "Set temperature to 0.8"),
        (["--max-tokens", "1024"], "Set max_tokens to 1024"),
        (["--top-p", "0.9"], "Set top_p to 0.9"),
        (["--frequency-penalty", "0.3"], "Set frequency_penalty to 0.3"),
        (["--stream"], "Enabled streaming"),
        (["--no-stream"], "Disabled streaming"),
        (["--save-media"], "Enabled save-media"),
        (["--no-save-media"], "Disabled save-media"),
        (["--media-dir", "/tmp/test"], "Set media_dir to '/tmp/test'"),
    ]

    for args, expected in tests:
        result = runner.invoke(load_cliver, ["session", "option", "set"] + args, obj=cliver_instance)
        assert result.exit_code == 0, f"Failed for {args}: {result.output}"
        assert expected in result.output, f"Expected '{expected}' in output for {args}: {result.output}"


def test_additional_options(load_cliver, init_config, simple_llm_model):
    """Additional inference options via --option flag."""
    runner = CliRunner()
    cliver_instance = _setup(runner, load_cliver)

    result = runner.invoke(
        load_cliver,
        ["session", "option", "set", "--option", "presence_penalty=0.5", "--option", 'stop=["\\n", "###"]'],
        obj=cliver_instance,
    )
    assert result.exit_code == 0
    assert "Updated additional options:" in result.output
    assert "presence_penalty" in result.output


def test_set_no_args_displays_options(load_cliver, init_config, simple_llm_model):
    """/session option set with no options displays current values."""
    runner = CliRunner()
    cliver_instance = _setup(runner, load_cliver)

    result = runner.invoke(load_cliver, ["session", "option", "set"], obj=cliver_instance)
    assert result.exit_code == 0
    assert "Session options:" in result.output
