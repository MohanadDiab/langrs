"""Tiling and windowing strategies for image processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union
from PIL import Image
import numpy as np


@dataclass
class Tile:
    """
    Represents a single tile/window from an image.
    
    Attributes:
        image: The tile image (PIL Image or numpy array)
        x_min: Minimum x coordinate in original image
        y_min: Minimum y coordinate in original image
        x_max: Maximum x coordinate in original image
        y_max: Maximum y coordinate in original image
        offset_x: Horizontal offset from original image (usually same as x_min)
        offset_y: Vertical offset from original image (usually same as y_min)
    """

    image: Union[Image.Image, np.ndarray]
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    offset_x: int = 0
    offset_y: int = 0

    def __post_init__(self):
        """Set default offsets if not provided."""
        if self.offset_x == 0:
            self.offset_x = self.x_min
        if self.offset_y == 0:
            self.offset_y = self.y_min


class TilingStrategy(ABC):
    """
    Abstract base class for tiling strategies.
    
    Different tiling strategies can be implemented for various
    image processing needs (sliding window, grid, etc.).
    """

    @abstractmethod
    def create_tiles(
        self,
        image: Union[Image.Image, np.ndarray],
        window_size: int,
        overlap: int,
    ) -> List[Tile]:
        """
        Create tiles from image.
        
        Args:
            image: Input image as PIL Image or numpy array
            window_size: Size of each tile/window
            overlap: Overlap size between adjacent tiles
            
        Returns:
            List of Tile objects
        """
        pass


class SlidingWindowTiler(TilingStrategy):
    """
    Sliding window tiling implementation.
    
    Creates overlapping tiles using a sliding window approach.
    Useful for processing large images in chunks.
    """

    def create_tiles(
        self,
        image: Union[Image.Image, np.ndarray],
        window_size: int,
        overlap: int,
    ) -> List[Tile]:
        """
        Create overlapping tiles using sliding window.
        
        Args:
            image: Input image as PIL Image or numpy array
            window_size: Size of each tile (square tiles assumed)
            overlap: Overlap size between adjacent tiles
            
        Returns:
            List of Tile objects
            
        Raises:
            ValueError: If window_size <= 0 or overlap >= window_size
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= window_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than window_size ({window_size})"
            )

        # Get image dimensions
        if isinstance(image, Image.Image):
            width, height = image.size
            is_pil = True
        else:
            height, width = image.shape[:2]
            is_pil = False

        tiles = []
        step = window_size - overlap

        # Create tiles
        for y in range(0, height, step):
            for x in range(0, width, step):
                x_min = x
                y_min = y
                x_max = min(x + window_size, width)
                y_max = min(y + window_size, height)

                # Extract tile
                if is_pil:
                    tile_image = image.crop((x_min, y_min, x_max, y_max))
                else:
                    tile_image = image[y_min:y_max, x_min:x_max]

                tiles.append(
                    Tile(
                        image=tile_image,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        offset_x=x_min,
                        offset_y=y_min,
                    )
                )

        return tiles
