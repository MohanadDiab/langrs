"""Tests for type definitions."""

import pytest
import numpy as np
import torch
from PIL import Image

from langrs.utils.types import (
    BoundingBox,
    BoundingBoxList,
    ImageInput,
    MaskArray,
    Device,
)


class TestTypeAliases:
    """Test type alias definitions."""

    def test_bounding_box_type(self):
        """Test BoundingBox type alias."""
        box: BoundingBox = (10.0, 20.0, 50.0, 60.0)
        assert len(box) == 4
        assert all(isinstance(x, float) for x in box)

    def test_bounding_box_list_type(self):
        """Test BoundingBoxList type alias."""
        boxes: BoundingBoxList = [
            (10.0, 20.0, 50.0, 60.0),
            (70.0, 80.0, 90.0, 100.0),
        ]
        assert len(boxes) == 2
        assert all(isinstance(box, tuple) for box in boxes)

    def test_image_input_pil(self):
        """Test ImageInput with PIL Image."""
        img: ImageInput = Image.new("RGB", (100, 100))
        assert isinstance(img, Image.Image)

    def test_image_input_numpy(self):
        """Test ImageInput with numpy array."""
        img: ImageInput = np.zeros((100, 100, 3), dtype=np.uint8)
        assert isinstance(img, np.ndarray)

    def test_image_input_str(self):
        """Test ImageInput with string path."""
        img: ImageInput = "path/to/image.jpg"
        assert isinstance(img, str)

    def test_mask_array_type(self):
        """Test MaskArray type alias."""
        mask: MaskArray = np.zeros((100, 100), dtype=np.uint8)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8

    def test_device_str(self):
        """Test Device type with string."""
        device: Device = "cpu"
        assert isinstance(device, str)

    def test_device_torch_device(self):
        """Test Device type with torch.device."""
        device: Device = torch.device("cpu")
        assert isinstance(device, torch.device)
