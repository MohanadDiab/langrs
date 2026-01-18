"""Tests for GroundingDINO detection model."""

import pytest
import numpy as np
from PIL import Image
import torch

from langrs.models.detection.grounding_dino import (
    GroundingDINODetector,
    GROUNDINGDINO_AVAILABLE,
)
from langrs.utils.exceptions import ModelLoadError, DetectionError


@pytest.mark.skipif(
    not GROUNDINGDINO_AVAILABLE,
    reason="groundingdino-py not installed",
)
class TestGroundingDINODetector:
    """Test GroundingDINO detector implementation."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = GroundingDINODetector()
        assert detector.model_variant == GroundingDINODetector.DEFAULT_VARIANT
        assert detector.device == torch.device("cpu")  # CPU-only env
        assert not detector.is_loaded

    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        detector = GroundingDINODetector(device="cpu")
        assert detector.device == torch.device("cpu")

    def test_initialization_with_variant(self):
        """Test initialization with model variant."""
        detector = GroundingDINODetector(model_variant="swint_ogc")
        assert detector.model_variant == "swint_ogc"

    def test_not_loaded_error(self, sample_image):
        """Test that detect raises error if model not loaded."""
        detector = GroundingDINODetector()
        with pytest.raises(DetectionError, match="Model not loaded"):
            detector.detect(sample_image, "test prompt")

    @pytest.mark.slow
    def test_load_weights_from_hf(self):
        """Test loading weights from Hugging Face."""
        detector = GroundingDINODetector()
        # This will download model, so mark as slow
        detector.load_weights()
        assert detector.is_loaded
        assert detector._model is not None

    def test_load_weights_invalid_path(self):
        """Test loading weights from invalid path."""
        detector = GroundingDINODetector()
        with pytest.raises(ModelLoadError, match="Model checkpoint not found"):
            detector.load_weights("nonexistent/path.pth")

    @pytest.mark.slow
    def test_detect_with_pil_image(self, sample_image):
        """Test detection with PIL Image."""
        detector = GroundingDINODetector()
        detector.load_weights()
        
        boxes = detector.detect(sample_image, "red object", box_threshold=0.3, text_threshold=0.3)
        assert isinstance(boxes, list)
        # May return empty list if no objects detected

    @pytest.mark.slow
    def test_detect_with_numpy_array(self, sample_numpy_image):
        """Test detection with numpy array."""
        detector = GroundingDINODetector()
        detector.load_weights()
        
        boxes = detector.detect(sample_numpy_image, "object", box_threshold=0.3, text_threshold=0.3)
        assert isinstance(boxes, list)

    @pytest.mark.slow
    def test_detect_with_thresholds(self, sample_image):
        """Test detection with custom thresholds."""
        detector = GroundingDINODetector()
        detector.load_weights()
        
        boxes = detector.detect(
            sample_image, "test", box_threshold=0.5, text_threshold=0.5
        )
        assert isinstance(boxes, list)

    def test_device_property(self):
        """Test device property."""
        detector = GroundingDINODetector(device="cpu")
        assert detector.device == torch.device("cpu")

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        detector = GroundingDINODetector()
        assert not detector.is_loaded


@pytest.mark.skipif(
    GROUNDINGDINO_AVAILABLE,
    reason="groundingdino-py is installed",
)
class TestGroundingDINODetectorNotAvailable:
    """Test behavior when groundingdino-py is not available."""

    def test_import_error(self):
        """Test that ImportError is raised when not available."""
        with pytest.raises(ImportError, match="groundingdino-py is not installed"):
            GroundingDINODetector()
