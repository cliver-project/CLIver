"""Tests for resolve_secret() three-layer resolution chain."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def key_store():
    from cliver.key_store import KeyStore

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        ks.set("openai_key", "sk-from-keystore")
        ks.set("MY_SPECIAL", "from-ks-not-env")
        yield ks


def test_resolve_from_keystore(key_store):
    from cliver.template_utils import resolve_secret

    assert resolve_secret("openai_key", key_store) == "sk-from-keystore"


def test_resolve_from_env_var():
    from cliver.key_store import KeyStore
    from cliver.template_utils import resolve_secret

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
            assert resolve_secret("OPENAI_API_KEY", ks) == "sk-from-env"


def test_resolve_literal():
    from cliver.key_store import KeyStore
    from cliver.template_utils import resolve_secret

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        assert resolve_secret("sk-hardcoded-123", ks) == "sk-hardcoded-123"


def test_keystore_takes_priority_over_env(key_store):
    from cliver.template_utils import resolve_secret

    with patch.dict(os.environ, {"MY_SPECIAL": "from-env"}):
        assert resolve_secret("MY_SPECIAL", key_store) == "from-ks-not-env"


def test_env_var_only_for_all_uppercase():
    from cliver.key_store import KeyStore
    from cliver.template_utils import resolve_secret

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        with patch.dict(os.environ, {"mixedCase": "should-not-match"}):
            assert resolve_secret("mixedCase", ks) == "mixedCase"


def test_env_var_with_underscores():
    from cliver.key_store import KeyStore
    from cliver.template_utils import resolve_secret

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        with patch.dict(os.environ, {"MY_API_KEY": "sk-env-val"}):
            assert resolve_secret("MY_API_KEY", ks) == "sk-env-val"


def test_env_var_not_set_falls_to_literal():
    from cliver.key_store import KeyStore
    from cliver.template_utils import resolve_secret

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        assert resolve_secret("NONEXISTENT_VAR_XYZ", ks) == "NONEXISTENT_VAR_XYZ"


def test_resolve_empty_string():
    from cliver.key_store import KeyStore
    from cliver.template_utils import resolve_secret

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        assert resolve_secret("", ks) == ""


def test_resolve_none_safe():
    from cliver.key_store import KeyStore
    from cliver.template_utils import resolve_secret

    with tempfile.TemporaryDirectory() as tmpdir:
        ks = KeyStore(Path(tmpdir) / "keys.db")
        assert resolve_secret(None, ks) is None


def test_render_template_still_works():
    from cliver.template_utils import render_template_if_needed

    with patch.dict(os.environ, {"MY_VAR": "hello"}):
        assert render_template_if_needed("prefix-{{ env.MY_VAR }}-suffix") == "prefix-hello-suffix"


def test_keyring_removed_from_jinja():
    from cliver.template_utils import get_jinja_env

    env = get_jinja_env()
    assert "keyring" not in env.globals
