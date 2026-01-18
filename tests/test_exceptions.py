"""Tests for custom exception hierarchy."""

import pytest

from langrs.utils.exceptions import (
    LangRSError,
    ModelLoadError,
    ImageLoadError,
    DetectionError,
    SegmentationError,
    ConfigurationError,
)


class TestLangRSError:
    """Test base exception class."""

    def test_base_exception_creation(self):
        """Test creating base exception."""
        error = LangRSError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_base_exception_with_args(self):
        """Test base exception with additional args."""
        error = LangRSError("Test error", "arg1", "arg2")
        assert error.message == "Test error"


class TestModelLoadError:
    """Test ModelLoadError exception."""

    def test_model_load_error_creation(self):
        """Test creating ModelLoadError."""
        error = ModelLoadError("Failed to load model")
        assert isinstance(error, LangRSError)
        assert str(error) == "Failed to load model"


class TestImageLoadError:
    """Test ImageLoadError exception."""

    def test_image_load_error_creation(self):
        """Test creating ImageLoadError."""
        error = ImageLoadError("Failed to load image")
        assert isinstance(error, LangRSError)
        assert str(error) == "Failed to load image"


class TestDetectionError:
    """Test DetectionError exception."""

    def test_detection_error_creation(self):
        """Test creating DetectionError."""
        error = DetectionError("Detection failed")
        assert isinstance(error, LangRSError)
        assert str(error) == "Detection failed"


class TestSegmentationError:
    """Test SegmentationError exception."""

    def test_segmentation_error_creation(self):
        """Test creating SegmentationError."""
        error = SegmentationError("Segmentation failed")
        assert isinstance(error, LangRSError)
        assert str(error) == "Segmentation failed"


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_configuration_error_creation(self):
        """Test creating ConfigurationError."""
        error = ConfigurationError("Invalid configuration")
        assert isinstance(error, LangRSError)
        assert str(error) == "Invalid configuration"
