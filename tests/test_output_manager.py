"""Tests for output manager module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from langrs.io.output_manager import OutputManager


class TestOutputManager:
    """Test OutputManager class."""

    def test_initialization_timestamped(self):
        """Test initialization with timestamped directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir, create_timestamped=True)
            assert manager.base_path == Path(tmpdir)
            assert manager.output_path.exists()
            assert manager.output_path.is_dir()
            # Check it has timestamp format
            assert manager.output_path.name.startswith("20")  # Year

    def test_initialization_not_timestamped(self):
        """Test initialization without timestamped directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir, create_timestamped=False)
            assert manager.base_path == Path(tmpdir)
            assert manager.output_path == Path(tmpdir)
            assert manager.output_path.exists()

    def test_get_path(self):
        """Test getting file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir, create_timestamped=False)
            path = manager.get_path("test.jpg")
            assert isinstance(path, Path)
            assert path.parent == Path(tmpdir)
            assert path.name == "test.jpg"

    def test_get_path_str(self):
        """Test getting file path as string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir, create_timestamped=False)
            path_str = manager.get_path_str("test.jpg")
            assert isinstance(path_str, str)
            assert path_str.endswith("test.jpg")

    def test_base_dir_property(self):
        """Test base_dir property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            assert manager.base_dir == Path(tmpdir)

    def test_output_dir_property(self):
        """Test output_dir property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            assert manager.output_dir.exists()
            assert manager.output_dir.is_dir()

    def test_multiple_paths(self):
        """Test getting multiple paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir, create_timestamped=False)
            path1 = manager.get_path("file1.jpg")
            path2 = manager.get_path("file2.jpg")
            assert path1 != path2
            assert path1.name == "file1.jpg"
            assert path2.name == "file2.jpg"

    def test_nested_paths(self):
        """Test paths with subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir, create_timestamped=False)
            path = manager.get_path("subdir/file.jpg")
            assert "subdir" in str(path)
            assert path.name == "file.jpg"
