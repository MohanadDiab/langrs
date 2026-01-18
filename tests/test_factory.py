"""Tests for model factory."""

import pytest
import torch

from langrs.models.factory import ModelFactory
from langrs.models.registry import ModelRegistry
from langrs.utils.exceptions import ModelLoadError


class TestModelFactory:
    """Test ModelFactory class."""

    def test_create_detection_model_unknown(self):
        """Test creating unknown detection model raises error."""
        with pytest.raises(ModelLoadError, match="Unknown detection model"):
            ModelFactory.create_detection_model("unknown_model")

    def test_create_segmentation_model_unknown(self):
        """Test creating unknown segmentation model raises error."""
        with pytest.raises(ModelLoadError, match="Unknown segmentation model"):
            ModelFactory.create_segmentation_model("unknown_model")

    def test_create_detection_model_with_device(self):
        """Test creating detection model with device."""
        try:
            model = ModelFactory.create_detection_model(
                "grounding_dino", device="cpu"
            )
            assert model.device == torch.device("cpu")
        except (ImportError, ModelLoadError):
            pytest.skip("groundingdino-py not available")

    def test_create_segmentation_model_with_device(self):
        """Test creating segmentation model with device."""
        try:
            model = ModelFactory.create_segmentation_model("sam", device="cpu")
            assert model.device == torch.device("cpu")
        except (ImportError, ModelLoadError):
            pytest.skip("segment-anything-py not available")
