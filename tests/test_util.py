import tempfile
from pathlib import Path

from cliver.util import read_context_files


def test_read_context_files_default():
    """Test that read_context_files works with default parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Cliver.md file
        cliver_md_path = Path(tmpdir) / "Cliver.md"
        cliver_md_path.write_text("# Test Content\nThis is test content.")

        # Test with default parameters
        context = read_context_files(tmpdir)
        assert "Content from Cliver.md" in context
        assert "This is test content." in context


def test_read_context_files_custom_filter():
    """Test that read_context_files works with custom file filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        readme_path = Path(tmpdir) / "README.md"
        readme_path.write_text("# README\nThis is the README content.")

        custom_path = Path(tmpdir) / "CUSTOM.md"
        custom_path.write_text("# Custom\nThis is custom content.")

        # Test with custom filter
        context = read_context_files(tmpdir, ["README.md", "CUSTOM.md"])
        assert "Content from README.md" in context
        assert "This is the README content." in context
        assert "Content from CUSTOM.md" in context
        assert "This is custom content." in context


def test_read_context_files_mixed_existing_missing():
    """Test that read_context_files handles mixed existing/missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create only one of the requested files
        readme_path = Path(tmpdir) / "README.md"
        readme_path.write_text("# README\nThis is the README content.")

        # CUSTOM.md is not created, should be skipped

        # Test with mixed files
        context = read_context_files(tmpdir, ["README.md", "CUSTOM.md"])
        assert "Content from README.md" in context
        assert "This is the README content." in context
        # CUSTOM.md should not be in the context since it doesn't exist
        assert "Content from CUSTOM.md" not in context


def test_read_context_files_default_first_match_only():
    """Default mode reads only the first match (Cliver.md preferred over CLAUDE.md)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "Cliver.md").write_text("cliver content")
        Path(tmpdir, "CLAUDE.md").write_text("claude content")

        context = read_context_files(tmpdir)
        assert "cliver content" in context
        assert "claude content" not in context


def test_read_context_files_default_fallback_to_claude():
    """When Cliver.md is absent, default mode falls back to CLAUDE.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "CLAUDE.md").write_text("claude fallback")

        context = read_context_files(tmpdir)
        assert "Content from CLAUDE.md" in context
        assert "claude fallback" in context


def test_read_context_files_truncation():
    """Content exceeding max_chars is truncated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "Cliver.md").write_text("A" * 5000)

        context = read_context_files(tmpdir, max_chars=100)
        assert "...(truncated)" in context
        # Header + 100 chars + truncation marker
        assert "A" * 100 in context
        assert "A" * 101 not in context
