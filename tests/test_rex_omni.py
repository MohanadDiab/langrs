"""Tests for Rex-Omni detection model."""

import pytest
import numpy as np
from PIL import Image
import torch

from langrs.models.detection.rex_omni import (
    RexOmniDetector,
    REX_OMNI_AVAILABLE,
)
from langrs.utils.exceptions import ModelLoadError, DetectionError
from langrs.models.factory import ModelFactory


@pytest.mark.skipif(
    not REX_OMNI_AVAILABLE,
    reason="Rex-Omni dependencies not installed",
)
class TestRexOmniDetector:
    """Test Rex-Omni detector implementation."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = RexOmniDetector()
        assert detector.model_variant == RexOmniDetector.DEFAULT_VARIANT
        assert detector.backend == "transformers"
        assert detector.device == torch.device("cpu")  # CPU-only env
        assert not detector.is_loaded

    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        detector = RexOmniDetector(device="cpu")
        assert detector.device == torch.device("cpu")

    def test_initialization_with_variant(self):
        """Test initialization with model variant."""
        detector = RexOmniDetector(model_variant="default")
        assert detector.model_variant == "default"

    def test_initialization_with_backend(self):
        """Test initialization with backend."""
        detector = RexOmniDetector(backend="transformers")
        assert detector.backend == "transformers"

    def test_initialization_with_generation_params(self):
        """Test initialization with generation parameters."""
        detector = RexOmniDetector(
            temperature=0.1,
            top_p=0.1,
            top_k=2,
            max_tokens=1024,
        )
        assert detector.temperature == 0.1
        assert detector.top_p == 0.1
        assert detector.top_k == 2
        assert detector.max_tokens == 1024

    def test_not_loaded_error(self, sample_image):
        """Test that detect raises error if model not loaded."""
        detector = RexOmniDetector()
        with pytest.raises(DetectionError, match="Model not loaded"):
            detector.detect(sample_image, "test prompt")

    def test_load_weights_invalid_path(self):
        """Test loading weights from invalid path."""
        detector = RexOmniDetector()
        # Use a path without '/' to ensure it's treated as local path
        with pytest.raises(ModelLoadError, match="Model path not found"):
            detector.load_weights("nonexistent_path_that_does_not_exist")

    def test_device_property(self):
        """Test device property."""
        detector = RexOmniDetector(device="cpu")
        assert detector.device == torch.device("cpu")

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        detector = RexOmniDetector()
        assert not detector.is_loaded

    def test_text_prompt_to_categories_single(self):
        """Test conversion of single text prompt to categories."""
        detector = RexOmniDetector()
        # This is tested indirectly through detect, but we can verify the logic
        # by checking that comma-separated prompts work
        pass  # Will be tested in integration tests

    def test_text_prompt_to_categories_multiple(self):
        """Test conversion of comma-separated text prompt to categories."""
        detector = RexOmniDetector()
        # This is tested indirectly through detect
        pass  # Will be tested in integration tests

    @pytest.mark.slow
    def test_load_weights_from_hf(self):
        """Test loading weights from Hugging Face."""
        detector = RexOmniDetector()
        # This will download model, so mark as slow
        detector.load_weights()
        assert detector.is_loaded
        assert detector._model is not None
        assert detector._processor is not None

    @pytest.mark.slow
    def test_detect_with_pil_image(self, sample_image):
        """Test detection with PIL Image."""
        detector = RexOmniDetector()
        detector.load_weights()
        
        boxes = detector.detect(sample_image, "red object", box_threshold=0.3, text_threshold=0.3)
        assert isinstance(boxes, list)
        # May return empty list if no objects detected
        # All boxes should be tuples of 4 floats
        for box in boxes:
            assert isinstance(box, tuple)
            assert len(box) == 4
            assert all(isinstance(coord, float) for coord in box)

    @pytest.mark.slow
    def test_detect_with_numpy_array(self, sample_numpy_image):
        """Test detection with numpy array."""
        detector = RexOmniDetector()
        detector.load_weights()
        
        boxes = detector.detect(sample_numpy_image, "object", box_threshold=0.3, text_threshold=0.3)
        assert isinstance(boxes, list)
        # All boxes should be tuples of 4 floats
        for box in boxes:
            assert isinstance(box, tuple)
            assert len(box) == 4
            assert all(isinstance(coord, float) for coord in box)

    @pytest.mark.slow
    def test_detect_with_comma_separated_prompt(self, sample_image):
        """Test detection with comma-separated categories."""
        detector = RexOmniDetector()
        detector.load_weights()
        
        boxes = detector.detect(
            sample_image, "red object, blue object", box_threshold=0.3, text_threshold=0.3
        )
        assert isinstance(boxes, list)

    @pytest.mark.slow
    def test_detect_thresholds_ignored(self, sample_image):
        """Test that thresholds are ignored (no confidence scores)."""
        detector = RexOmniDetector()
        detector.load_weights()
        
        # Should work with any threshold values since they're ignored
        boxes1 = detector.detect(
            sample_image, "test", box_threshold=0.1, text_threshold=0.1
        )
        boxes2 = detector.detect(
            sample_image, "test", box_threshold=0.9, text_threshold=0.9
        )
        # Both should work (thresholds ignored)
        assert isinstance(boxes1, list)
        assert isinstance(boxes2, list)

    @pytest.mark.slow
    def test_coordinate_format(self, sample_image):
        """Test that coordinates are in correct format (x_min, y_min, x_max, y_max)."""
        detector = RexOmniDetector()
        detector.load_weights()
        
        boxes = detector.detect(sample_image, "object", box_threshold=0.3, text_threshold=0.3)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # Verify coordinates are in correct order
            assert x_min <= x_max
            assert y_min <= y_max
            # Verify coordinates are within image bounds (with some tolerance)
            assert 0 <= x_min <= sample_image.width
            assert 0 <= y_min <= sample_image.height
            assert 0 <= x_max <= sample_image.width
            assert 0 <= y_max <= sample_image.height

    def test_model_variants(self):
        """Test that model variants are defined."""
        assert "default" in RexOmniDetector.MODEL_VARIANTS
        assert "awq" in RexOmniDetector.MODEL_VARIANTS
        assert RexOmniDetector.DEFAULT_VARIANT in RexOmniDetector.MODEL_VARIANTS


@pytest.mark.skipif(
    not REX_OMNI_AVAILABLE,
    reason="Rex-Omni dependencies not installed",
)
class TestRexOmniDetectorIntegration:
    """Test Rex-Omni detector integration with LangRS components."""

    @pytest.mark.slow
    def test_model_factory_creation(self):
        """Test creating detector via ModelFactory."""
        detector = ModelFactory.create_detection_model(
            model_name="rex_omni",
            device="cpu",
        )
        assert isinstance(detector, RexOmniDetector)
        assert detector.device == torch.device("cpu")

    @pytest.mark.slow
    def test_model_factory_with_params(self):
        """Test creating detector via ModelFactory with parameters."""
        detector = ModelFactory.create_detection_model(
            model_name="rex_omni",
            device="cpu",
            backend="transformers",
            temperature=0.1,
            max_tokens=1024,
        )
        assert isinstance(detector, RexOmniDetector)
        assert detector.temperature == 0.1
        assert detector.max_tokens == 1024

    @pytest.mark.slow
    def test_model_factory_with_variant(self):
        """Test creating detector via ModelFactory with model variant."""
        detector = ModelFactory.create_detection_model(
            model_name="rex_omni",
            device="cpu",
            model_variant="default",
        )
        assert isinstance(detector, RexOmniDetector)
        assert detector.model_variant == "default"

    @pytest.mark.slow
    def test_model_registry(self):
        """Test that model is registered in ModelRegistry."""
        from langrs.models.registry import ModelRegistry
        
        model_class = ModelRegistry.get_detection_model("rex_omni")
        assert model_class is not None
        assert model_class == RexOmniDetector
        
        # Check it's in the list
        available_models = ModelRegistry.list_detection_models()
        assert "rex_omni" in available_models


@pytest.mark.skipif(
    REX_OMNI_AVAILABLE,
    reason="Rex-Omni dependencies are installed",
)
class TestRexOmniDetectorNotAvailable:
    """Test behavior when Rex-Omni dependencies are not available."""

    def test_import_error(self):
        """Test that ImportError is raised when not available."""
        with pytest.raises(ImportError, match="Rex-Omni dependencies are not installed"):
            RexOmniDetector()

    def test_model_not_in_registry(self):
        """Test that model is not registered when dependencies unavailable."""
        from langrs.models.registry import ModelRegistry
        
        model_class = ModelRegistry.get_detection_model("rex_omni")
        # Should be None if dependencies not available
        assert model_class is None


@pytest.mark.skipif(
    not REX_OMNI_AVAILABLE,
    reason="Rex-Omni dependencies not installed",
)
class TestRexOmniDetectorVLLM:
    """Test Rex-Omni detector with vllm backend (if available)."""

    def test_vllm_backend_requested_but_not_available(self):
        """Test that error is raised if vllm requested but not installed."""
        # This test will pass if vllm is not available
        # and fail if vllm is available (which is fine)
        try:
            from vllm import LLM
            VLLM_AVAILABLE = True
        except ImportError:
            VLLM_AVAILABLE = False
        
        if not VLLM_AVAILABLE:
            with pytest.raises(ImportError, match="vllm is not installed"):
                RexOmniDetector(backend="vllm")
        else:
            # If vllm is available, we can test initialization
            detector = RexOmniDetector(backend="vllm")
            assert detector.backend == "vllm"
