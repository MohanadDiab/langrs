"""Tests for image loading module."""

import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from langrs.processing.image_loader import ImageLoader, ImageData
from langrs.utils.exceptions import ImageLoadError


class TestImageData:
    """Test ImageData dataclass."""

    def test_image_data_creation(self, sample_image):
        """Test creating ImageData."""
        data = ImageData(
            image_path="test.jpg",
            pil_image=sample_image,
            np_image=np.array(sample_image),
            source_crs="EPSG:32633",
            transform=None,
            is_georeferenced=True,
        )
        assert data.image_path == "test.jpg"
        assert data.is_georeferenced is True

    def test_image_data_auto_georeferenced(self, sample_image):
        """Test is_georeferenced is set automatically."""
        data = ImageData(
            image_path="test.jpg",
            pil_image=sample_image,
            np_image=np.array(sample_image),
            source_crs=None,
            transform=None,
            is_georeferenced=False,
        )
        assert data.is_georeferenced is False


class TestImageLoader:
    """Test ImageLoader class."""

    def test_load_from_pil_image(self, sample_image):
        """Test loading from PIL Image."""
        data = ImageLoader.load(sample_image)
        assert isinstance(data, ImageData)
        assert data.image_path is None
        assert isinstance(data.pil_image, Image.Image)
        assert isinstance(data.np_image, np.ndarray)
        assert not data.is_georeferenced

    def test_load_from_numpy_array(self, sample_numpy_image):
        """Test loading from numpy array."""
        data = ImageLoader.load(sample_numpy_image)
        assert isinstance(data, ImageData)
        assert data.image_path is None
        assert isinstance(data.pil_image, Image.Image)
        assert isinstance(data.np_image, np.ndarray)

    def test_load_from_numpy_array_with_alpha(self):
        """Test loading numpy array with alpha channel."""
        rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        data = ImageLoader.load(rgba_image)
        assert data.np_image.shape == (100, 100, 3)  # Alpha dropped

    def test_load_from_numpy_array_float(self):
        """Test loading numpy array with float values."""
        float_image = np.random.rand(100, 100, 3).astype(np.float32)
        data = ImageLoader.load(float_image)
        assert data.np_image.dtype == np.uint8
        assert np.all(data.np_image <= 255)

    def test_load_from_file_jpg(self):
        """Test loading from JPG file."""
        # Create temporary JPG
        img = Image.new("RGB", (100, 100), color="red")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            jpg_path = f.name

        try:
            data = ImageLoader.load(jpg_path)
            assert data.image_path == jpg_path
            assert isinstance(data.pil_image, Image.Image)
            assert not data.is_georeferenced
        finally:
            Path(jpg_path).unlink()

    def test_load_from_file_png(self):
        """Test loading from PNG file."""
        img = Image.new("RGB", (100, 100), color="blue")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            png_path = f.name

        try:
            data = ImageLoader.load(png_path)
            assert data.image_path == png_path
        finally:
            Path(png_path).unlink()

    def test_load_invalid_file(self):
        """Test loading non-existent file."""
        with pytest.raises(ImageLoadError, match="Image file not found"):
            ImageLoader.load("nonexistent_file.jpg")

    def test_load_unsupported_format(self):
        """Test loading unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"not an image")
            xyz_path = f.name

        try:
            with pytest.raises(ImageLoadError, match="Unsupported image format"):
                ImageLoader.load(xyz_path)
        finally:
            Path(xyz_path).unlink()

    def test_load_invalid_array_shape(self):
        """Test loading invalid array shape."""
        invalid_array = np.random.rand(100, 100)  # 2D instead of 3D
        with pytest.raises(ImageLoadError, match="Expected RGB"):
            ImageLoader.load(invalid_array)

    def test_load_invalid_type(self):
        """Test loading invalid type."""
        with pytest.raises(ImageLoadError, match="Unsupported image input type"):
            ImageLoader.load(123)  # Not a valid type
