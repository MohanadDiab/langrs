"""Tests for abstract base model classes."""

import pytest
import numpy as np
import torch
from PIL import Image

from langrs.models.base import DetectionModel, SegmentationModel
from langrs.utils.exceptions import DetectionError, SegmentationError


class TestDetectionModel:
    """Test DetectionModel abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            DetectionModel()

    def test_concrete_implementation_required_methods(self):
        """Test that concrete implementation must implement all methods."""

        class IncompleteDetectionModel(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

        # Missing load_weights, device, is_loaded
        with pytest.raises(TypeError):
            IncompleteDetectionModel()

    def test_complete_implementation(self, sample_image):
        """Test complete implementation of DetectionModel."""

        class CompleteDetectionModel(DetectionModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return [(10.0, 10.0, 50.0, 50.0)]

            def load_weights(self, model_path=None):
                self._loaded = True

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = CompleteDetectionModel()
        assert not model.is_loaded
        assert model.device == torch.device("cpu")

        boxes = model.detect(sample_image, "test")
        assert len(boxes) == 1
        assert boxes[0] == (10.0, 10.0, 50.0, 50.0)

        model.load_weights()
        assert model.is_loaded

    def test_detect_with_pil_image(self, sample_image):
        """Test detect method accepts PIL Image."""

        class TestModel(DetectionModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                assert isinstance(image, Image.Image)
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = TestModel()
        model.detect(sample_image, "test")

    def test_detect_with_numpy_array(self, sample_numpy_image):
        """Test detect method accepts numpy array."""

        class TestModel(DetectionModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                assert isinstance(image, np.ndarray)
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = TestModel()
        model.detect(sample_numpy_image, "test")

    def test_detect_with_thresholds(self, sample_image):
        """Test detect method with custom thresholds."""

        class TestModel(DetectionModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False
                self.last_box_threshold = None
                self.last_text_threshold = None

            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                self.last_box_threshold = box_threshold
                self.last_text_threshold = text_threshold
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = TestModel()
        model.detect(sample_image, "test", box_threshold=0.5, text_threshold=0.7)
        assert model.last_box_threshold == 0.5
        assert model.last_text_threshold == 0.7


class TestSegmentationModel:
    """Test SegmentationModel abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            SegmentationModel()

    def test_concrete_implementation_required_methods(self):
        """Test that concrete implementation must implement all methods."""

        class IncompleteSegmentationModel(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((1, 10, 10))

        # Missing load_weights, device, is_loaded
        with pytest.raises(TypeError):
            IncompleteSegmentationModel()

    def test_complete_implementation(self, sample_image, sample_boxes_tensor):
        """Test complete implementation of SegmentationModel."""

        class CompleteSegmentationModel(SegmentationModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def segment(self, image, boxes):
                n = boxes.shape[0]
                return torch.zeros((n, 100, 100), dtype=torch.bool)

            def load_weights(self, model_path=None):
                self._loaded = True

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = CompleteSegmentationModel()
        assert not model.is_loaded
        assert model.device == torch.device("cpu")

        masks = model.segment(sample_image, sample_boxes_tensor)
        assert masks.shape[0] == sample_boxes_tensor.shape[0]
        assert masks.shape[1] == 100
        assert masks.shape[2] == 100

        model.load_weights()
        assert model.is_loaded

    def test_segment_with_pil_image(self, sample_image, sample_boxes_tensor):
        """Test segment method accepts PIL Image."""

        class TestModel(SegmentationModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def segment(self, image, boxes):
                assert isinstance(image, Image.Image)
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = TestModel()
        model.segment(sample_image, sample_boxes_tensor)

    def test_segment_with_numpy_array(self, sample_numpy_image, sample_boxes_tensor):
        """Test segment method accepts numpy array."""

        class TestModel(SegmentationModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def segment(self, image, boxes):
                assert isinstance(image, np.ndarray)
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = TestModel()
        model.segment(sample_numpy_image, sample_boxes_tensor)

    def test_set_image_optional(self, sample_image):
        """Test that set_image is optional and has default implementation."""

        class TestModel(SegmentationModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = TestModel()
        # Should not raise error
        model.set_image(sample_image)

    def test_segment_returns_correct_shape(self, sample_image, sample_boxes_tensor):
        """Test that segment returns masks with correct shape."""

        class TestModel(SegmentationModel):
            def __init__(self):
                self._device = torch.device("cpu")
                self._loaded = False

            def segment(self, image, boxes):
                n = boxes.shape[0]
                h, w = 50, 50
                return torch.zeros((n, h, w), dtype=torch.bool)

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return self._device

            @property
            def is_loaded(self):
                return self._loaded

        model = TestModel()
        masks = model.segment(sample_image, sample_boxes_tensor)
        assert masks.dtype == torch.bool
        assert masks.shape == (sample_boxes_tensor.shape[0], 50, 50)
