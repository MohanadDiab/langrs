"""Tests for SAM segmentation model."""

import pytest
import numpy as np
from PIL import Image
import torch

from langrs.models.segmentation.sam import SAMSegmenter, SAM_AVAILABLE
from langrs.utils.exceptions import ModelLoadError, SegmentationError


@pytest.mark.skipif(
    not SAM_AVAILABLE,
    reason="segment-anything-py not installed",
)
class TestSAMSegmenter:
    """Test SAM segmenter implementation."""

    def test_initialization(self):
        """Test segmenter initialization."""
        segmenter = SAMSegmenter()
        assert segmenter.model_variant == SAMSegmenter.DEFAULT_VARIANT
        assert segmenter.device == torch.device("cpu")  # CPU-only env
        assert not segmenter.is_loaded

    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        segmenter = SAMSegmenter(device="cpu")
        assert segmenter.device == torch.device("cpu")

    def test_initialization_with_variant(self):
        """Test initialization with model variant."""
        segmenter = SAMSegmenter(model_variant="vit_b")
        assert segmenter.model_variant == "vit_b"

    def test_not_loaded_error(self, sample_image, sample_boxes_tensor):
        """Test that segment raises error if model not loaded."""
        segmenter = SAMSegmenter()
        with pytest.raises(SegmentationError, match="Model not loaded"):
            segmenter.segment(sample_image, sample_boxes_tensor)

    @pytest.mark.slow
    def test_load_weights_from_hf(self):
        """Test loading weights from Hugging Face."""
        segmenter = SAMSegmenter(model_variant="vit_b")  # Use smaller model for testing
        # This will download model, so mark as slow
        segmenter.load_weights()
        assert segmenter.is_loaded
        assert segmenter._model is not None
        assert segmenter._predictor is not None

    def test_load_weights_invalid_path(self):
        """Test loading weights from invalid path."""
        segmenter = SAMSegmenter()
        with pytest.raises(ModelLoadError, match="Model checkpoint not found"):
            segmenter.load_weights("nonexistent/path.pth")

    @pytest.mark.slow
    def test_set_image(self, sample_image):
        """Test set_image method."""
        segmenter = SAMSegmenter(model_variant="vit_b")
        segmenter.load_weights()
        
        segmenter.set_image(sample_image)
        assert segmenter._current_image is not None

    @pytest.mark.slow
    def test_set_image_error_not_loaded(self, sample_image):
        """Test set_image raises error if model not loaded."""
        segmenter = SAMSegmenter()
        with pytest.raises(SegmentationError, match="Model not loaded"):
            segmenter.set_image(sample_image)

    @pytest.mark.slow
    def test_segment_with_pil_image(self, sample_image, sample_boxes_tensor):
        """Test segmentation with PIL Image."""
        segmenter = SAMSegmenter(model_variant="vit_b")
        segmenter.load_weights()
        
        masks = segmenter.segment(sample_image, sample_boxes_tensor)
        assert isinstance(masks, torch.Tensor)
        assert masks.shape[0] == sample_boxes_tensor.shape[0]
        assert masks.dtype == torch.float32

    @pytest.mark.slow
    def test_segment_with_numpy_array(self, sample_numpy_image, sample_boxes_tensor):
        """Test segmentation with numpy array."""
        segmenter = SAMSegmenter(model_variant="vit_b")
        segmenter.load_weights()
        
        masks = segmenter.segment(sample_numpy_image, sample_boxes_tensor)
        assert isinstance(masks, torch.Tensor)
        assert masks.shape[0] == sample_boxes_tensor.shape[0]

    @pytest.mark.slow
    def test_segment_returns_binary_masks(self, sample_image, sample_boxes_tensor):
        """Test that segment returns binary masks."""
        segmenter = SAMSegmenter(model_variant="vit_b")
        segmenter.load_weights()
        
        masks = segmenter.segment(sample_image, sample_boxes_tensor)
        # Masks should be binary (0 or 1)
        assert torch.all((masks == 0) | (masks == 1))

    def test_device_property(self):
        """Test device property."""
        segmenter = SAMSegmenter(device="cpu")
        assert segmenter.device == torch.device("cpu")

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        segmenter = SAMSegmenter()
        assert not segmenter.is_loaded


@pytest.mark.skipif(
    SAM_AVAILABLE,
    reason="segment-anything-py is installed",
)
class TestSAMSegmenterNotAvailable:
    """Test behavior when segment-anything-py is not available."""

    def test_import_error(self):
        """Test that ImportError is raised when not available."""
        with pytest.raises(ImportError, match="segment-anything-py is not installed"):
            SAMSegmenter()
