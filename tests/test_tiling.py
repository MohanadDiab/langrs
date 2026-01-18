"""Tests for tiling module."""

import pytest
import numpy as np
from PIL import Image

from langrs.processing.tiling import (
    TilingStrategy,
    SlidingWindowTiler,
    Tile,
)


class TestTile:
    """Test Tile dataclass."""

    def test_tile_creation(self, sample_image):
        """Test creating a Tile."""
        tile = Tile(
            image=sample_image,
            x_min=0,
            y_min=0,
            x_max=50,
            y_max=50,
        )
        assert tile.x_min == 0
        assert tile.y_min == 0
        assert tile.x_max == 50
        assert tile.y_max == 50
        assert tile.offset_x == 0  # Auto-set
        assert tile.offset_y == 0  # Auto-set

    def test_tile_with_offsets(self, sample_image):
        """Test Tile with custom offsets."""
        tile = Tile(
            image=sample_image,
            x_min=10,
            y_min=20,
            x_max=60,
            y_max=70,
            offset_x=10,
            offset_y=20,
        )
        assert tile.offset_x == 10
        assert tile.offset_y == 20


class TestSlidingWindowTiler:
    """Test SlidingWindowTiler implementation."""

    def test_create_tiles_pil_image(self, sample_image):
        """Test creating tiles from PIL Image."""
        tiler = SlidingWindowTiler()
        tiles = tiler.create_tiles(sample_image, window_size=50, overlap=10)

        assert len(tiles) > 0
        assert all(isinstance(tile, Tile) for tile in tiles)
        assert all(isinstance(tile.image, Image.Image) for tile in tiles)

    def test_create_tiles_numpy_array(self, sample_numpy_image):
        """Test creating tiles from numpy array."""
        tiler = SlidingWindowTiler()
        tiles = tiler.create_tiles(sample_numpy_image, window_size=50, overlap=10)

        assert len(tiles) > 0
        assert all(isinstance(tile, Tile) for tile in tiles)
        assert all(isinstance(tile.image, np.ndarray) for tile in tiles)

    def test_create_tiles_no_overlap(self, sample_image):
        """Test creating tiles with no overlap."""
        tiler = SlidingWindowTiler()
        tiles = tiler.create_tiles(sample_image, window_size=50, overlap=0)

        assert len(tiles) > 0
        # With 100x100 image and 50x50 windows, should get 4 tiles
        assert len(tiles) == 4

    def test_create_tiles_with_overlap(self, sample_image):
        """Test creating tiles with overlap."""
        tiler = SlidingWindowTiler()
        tiles = tiler.create_tiles(sample_image, window_size=50, overlap=25)

        assert len(tiles) > 0
        # With overlap, should get more tiles
        assert len(tiles) >= 4

    def test_tile_coordinates(self, sample_image):
        """Test that tile coordinates are correct."""
        tiler = SlidingWindowTiler()
        tiles = tiler.create_tiles(sample_image, window_size=50, overlap=0)

        # Check first tile
        first_tile = tiles[0]
        assert first_tile.x_min == 0
        assert first_tile.y_min == 0
        assert first_tile.x_max == 50
        assert first_tile.y_max == 50

    def test_invalid_window_size(self, sample_image):
        """Test invalid window size."""
        tiler = SlidingWindowTiler()
        with pytest.raises(ValueError, match="window_size must be positive"):
            tiler.create_tiles(sample_image, window_size=0, overlap=10)

        with pytest.raises(ValueError, match="window_size must be positive"):
            tiler.create_tiles(sample_image, window_size=-10, overlap=10)

    def test_invalid_overlap(self, sample_image):
        """Test invalid overlap."""
        tiler = SlidingWindowTiler()
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            tiler.create_tiles(sample_image, window_size=50, overlap=-10)

    def test_overlap_too_large(self, sample_image):
        """Test overlap >= window_size."""
        tiler = SlidingWindowTiler()
        with pytest.raises(ValueError, match="overlap.*must be less than window_size"):
            tiler.create_tiles(sample_image, window_size=50, overlap=50)

        with pytest.raises(ValueError, match="overlap.*must be less than window_size"):
            tiler.create_tiles(sample_image, window_size=50, overlap=60)

    def test_tiles_cover_image(self, sample_image):
        """Test that tiles cover the entire image."""
        tiler = SlidingWindowTiler()
        tiles = tiler.create_tiles(sample_image, window_size=50, overlap=10)

        # Check that tiles cover the image
        max_x = max(tile.x_max for tile in tiles)
        max_y = max(tile.y_max for tile in tiles)
        assert max_x >= 100  # Image width
        assert max_y >= 100  # Image height
