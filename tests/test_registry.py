"""Tests for model registry."""

import pytest

from langrs.models.registry import ModelRegistry
from langrs.models.base import DetectionModel, SegmentationModel
import torch


class TestModelRegistry:
    """Test ModelRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry.clear()

    def test_register_detection_model(self):
        """Test registering a detection model."""

        @ModelRegistry.register_detection("test_detector")
        class TestDetector(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        assert ModelRegistry.is_detection_model_registered("test_detector")
        assert "test_detector" in ModelRegistry.list_detection_models()

    def test_register_segmentation_model(self):
        """Test registering a segmentation model."""

        @ModelRegistry.register_segmentation("test_segmenter")
        class TestSegmenter(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        assert ModelRegistry.is_segmentation_model_registered("test_segmenter")
        assert "test_segmenter" in ModelRegistry.list_segmentation_models()

    def test_register_detection_model_not_subclass(self):
        """Test that non-DetectionModel cannot be registered."""

        with pytest.raises(TypeError, match="must inherit from DetectionModel"):

            @ModelRegistry.register_detection("invalid")
            class InvalidModel:
                pass

    def test_register_segmentation_model_not_subclass(self):
        """Test that non-SegmentationModel cannot be registered."""

        with pytest.raises(TypeError, match="must inherit from SegmentationModel"):

            @ModelRegistry.register_segmentation("invalid")
            class InvalidModel:
                pass

    def test_register_duplicate_detection_model(self):
        """Test that duplicate detection model names raise error."""

        @ModelRegistry.register_detection("duplicate")
        class FirstModel(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        with pytest.raises(ValueError, match="already registered"):

            @ModelRegistry.register_detection("duplicate")
            class SecondModel(DetectionModel):
                def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                    return []

                def load_weights(self, model_path=None):
                    pass

                @property
                def device(self):
                    return torch.device("cpu")

                @property
                def is_loaded(self):
                    return False

    def test_register_duplicate_segmentation_model(self):
        """Test that duplicate segmentation model names raise error."""

        @ModelRegistry.register_segmentation("duplicate")
        class FirstModel(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        with pytest.raises(ValueError, match="already registered"):

            @ModelRegistry.register_segmentation("duplicate")
            class SecondModel(SegmentationModel):
                def segment(self, image, boxes):
                    return torch.zeros((boxes.shape[0], 10, 10))

                def load_weights(self, model_path=None):
                    pass

                @property
                def device(self):
                    return torch.device("cpu")

                @property
                def is_loaded(self):
                    return False

    def test_get_detection_model(self):
        """Test retrieving a registered detection model."""

        @ModelRegistry.register_detection("test_detector")
        class TestDetector(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        model_class = ModelRegistry.get_detection_model("test_detector")
        assert model_class is TestDetector

    def test_get_segmentation_model(self):
        """Test retrieving a registered segmentation model."""

        @ModelRegistry.register_segmentation("test_segmenter")
        class TestSegmenter(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        model_class = ModelRegistry.get_segmentation_model("test_segmenter")
        assert model_class is TestSegmenter

    def test_get_nonexistent_detection_model(self):
        """Test retrieving non-existent detection model returns None."""
        assert ModelRegistry.get_detection_model("nonexistent") is None

    def test_get_nonexistent_segmentation_model(self):
        """Test retrieving non-existent segmentation model returns None."""
        assert ModelRegistry.get_segmentation_model("nonexistent") is None

    def test_list_detection_models(self):
        """Test listing all detection models."""

        @ModelRegistry.register_detection("model1")
        class Model1(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        @ModelRegistry.register_detection("model2")
        class Model2(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        models = ModelRegistry.list_detection_models()
        assert "model1" in models
        assert "model2" in models
        assert len(models) == 2

    def test_list_segmentation_models(self):
        """Test listing all segmentation models."""

        @ModelRegistry.register_segmentation("model1")
        class Model1(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        @ModelRegistry.register_segmentation("model2")
        class Model2(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        models = ModelRegistry.list_segmentation_models()
        assert "model1" in models
        assert "model2" in models
        assert len(models) == 2

    def test_is_detection_model_registered(self):
        """Test checking if detection model is registered."""

        @ModelRegistry.register_detection("registered")
        class Registered(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        assert ModelRegistry.is_detection_model_registered("registered")
        assert not ModelRegistry.is_detection_model_registered("not_registered")

    def test_is_segmentation_model_registered(self):
        """Test checking if segmentation model is registered."""

        @ModelRegistry.register_segmentation("registered")
        class Registered(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        assert ModelRegistry.is_segmentation_model_registered("registered")
        assert not ModelRegistry.is_segmentation_model_registered("not_registered")

    def test_unregister_detection_model(self):
        """Test unregistering a detection model."""

        @ModelRegistry.register_detection("temp")
        class Temp(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        assert ModelRegistry.is_detection_model_registered("temp")
        assert ModelRegistry.unregister_detection("temp")
        assert not ModelRegistry.is_detection_model_registered("temp")
        assert not ModelRegistry.unregister_detection("temp")  # Already removed

    def test_unregister_segmentation_model(self):
        """Test unregistering a segmentation model."""

        @ModelRegistry.register_segmentation("temp")
        class Temp(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        assert ModelRegistry.is_segmentation_model_registered("temp")
        assert ModelRegistry.unregister_segmentation("temp")
        assert not ModelRegistry.is_segmentation_model_registered("temp")
        assert not ModelRegistry.unregister_segmentation("temp")  # Already removed

    def test_clear_registry(self):
        """Test clearing all registered models."""

        @ModelRegistry.register_detection("det1")
        class Det1(DetectionModel):
            def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
                return []

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        @ModelRegistry.register_segmentation("seg1")
        class Seg1(SegmentationModel):
            def segment(self, image, boxes):
                return torch.zeros((boxes.shape[0], 10, 10))

            def load_weights(self, model_path=None):
                pass

            @property
            def device(self):
                return torch.device("cpu")

            @property
            def is_loaded(self):
                return False

        assert len(ModelRegistry.list_detection_models()) > 0
        assert len(ModelRegistry.list_segmentation_models()) > 0

        ModelRegistry.clear()

        assert len(ModelRegistry.list_detection_models()) == 0
        assert len(ModelRegistry.list_segmentation_models()) == 0
