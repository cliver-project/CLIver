"""Tests for sensitive file protection in read_file tool."""

import pytest

from cliver.tools.read_file import _is_sensitive_file


class TestSensitiveFileDetection:
    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            "/home/user/.env",
            ".env.local",
            ".env.production",
            "credentials.json",
            "/etc/shadow",
            "id_rsa",
            "/home/user/.ssh/id_rsa",
            "id_ed25519",
            "server.key",
            "cert.pem",
            "service_account.json",
            "service-account-key.json",
            ".netrc",
            ".pgpass",
            "token.json",
            "secrets.yaml",
            "secrets.yml",
            "app.p12",
            "keystore.jks",
            "htpasswd",
            "kubeconfig",
        ],
    )
    def test_sensitive_files_blocked(self, path):
        assert _is_sensitive_file(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "README.md",
            "config.yaml",
            "main.py",
            "environment.py",
            "credentials_test.py",
            "deploy.yaml",
            "Makefile",
            "package.json",
            ".gitignore",
            "secret_santa.py",
            "public_key.pub",
        ],
    )
    def test_safe_files_allowed(self, path):
        assert _is_sensitive_file(path) is False
