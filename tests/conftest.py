"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from PIL import Image
import torch

from langrs.models.base import DetectionModel, SegmentationModel
from langrs.core.config import LangRSConfig


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def sample_numpy_image():
    """Create a sample numpy array image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_boxes():
    """Create sample bounding boxes for testing."""
    return [(10.0, 10.0, 50.0, 50.0), (60.0, 60.0, 90.0, 90.0)]


@pytest.fixture
def sample_boxes_tensor(sample_boxes):
    """Create sample bounding boxes as tensor."""
    return torch.tensor(sample_boxes, dtype=torch.float32)


@pytest.fixture
def default_config():
    """Create a default LangRSConfig for testing."""
    return LangRSConfig()


@pytest.fixture
def mock_detection_model():
    """Create a mock detection model for testing."""
    class MockDetectionModel(DetectionModel):
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

    return MockDetectionModel


@pytest.fixture
def mock_segmentation_model():
    """Create a mock segmentation model for testing."""
    class MockSegmentationModel(SegmentationModel):
        def __init__(self):
            self._device = torch.device("cpu")
            self._loaded = False

        def segment(self, image, boxes):
            n = boxes.shape[0]
            h, w = 100, 100
            return torch.zeros((n, h, w), dtype=torch.bool)

        def load_weights(self, model_path=None):
            self._loaded = True

        @property
        def device(self):
            return self._device

        @property
        def is_loaded(self):
            return self._loaded

    return MockSegmentationModel
