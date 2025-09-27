import os
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