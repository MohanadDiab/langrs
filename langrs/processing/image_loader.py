"""Image loading and metadata extraction."""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from PIL import Image
import rasterio
from pathlib import Path
import os

from ..utils.exceptions import ImageLoadError


@dataclass
class ImageData:
    """
    Container for loaded image data and metadata.
    
    Attributes:
        image_path: Original file path, or None if not from file
        pil_image: Loaded image as PIL Image object
        np_image: Loaded image as numpy array (H, W, C)
        source_crs: Coordinate reference system if available (for GeoTIFF)
        transform: Geotransform matrix if available (for GeoTIFF)
        is_georeferenced: Whether the image has geospatial information
    """

    image_path: Optional[str]
    pil_image: Image.Image
    np_image: np.ndarray
    source_crs: Optional[str]
    transform: Optional[rasterio.transform.Affine]
    is_georeferenced: bool

    def __post_init__(self):
        """Set is_georeferenced based on source_crs."""
        self.is_georeferenced = self.source_crs is not None


class ImageLoader:
    """
    Handles image loading from various sources.
    
    Supports loading from:
    - File paths (GeoTIFF, PNG, JPG)
    - NumPy arrays
    - PIL Image objects
    """

    @staticmethod
    def load(source: Union[str, np.ndarray, Image.Image]) -> ImageData:
        """
        Load image from file path, numpy array, or PIL Image.
        
        Args:
            source: Image source as:
                   - File path (str): GeoTIFF, PNG, or JPG
                   - NumPy array: Shape (H, W, C) with 3 or 4 channels
                   - PIL Image: Any PIL Image object
                   
        Returns:
            ImageData object containing image and metadata
            
        Raises:
            ImageLoadError: If image loading fails
            FileNotFoundError: If file path does not exist
            ValueError: If image format is unsupported
            TypeError: If input type is not recognized
        """
        try:
            if isinstance(source, str):
                return ImageLoader._load_from_file(source)
            elif isinstance(source, np.ndarray):
                return ImageLoader._load_from_array(source)
            elif isinstance(source, Image.Image):
                return ImageLoader._load_from_pil(source)
            else:
                raise TypeError(
                    f"Unsupported image input type: {type(source)}. "
                    "Must be file path (str), numpy array, or PIL Image."
                )
        except (FileNotFoundError, ValueError, TypeError) as e:
            raise ImageLoadError(f"Failed to load image: {e}") from e
        except Exception as e:
            raise ImageLoadError(f"Unexpected error loading image: {e}") from e

    @staticmethod
    def _load_from_file(image_path: str) -> ImageData:
        """Load image from file path."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        ext = os.path.splitext(image_path)[-1].lower()
        transform = None
        source_crs = None

        if ext in [".tif", ".tiff"]:
            # Load GeoTIFF
            with rasterio.open(image_path) as src:
                # Read RGB channels (assuming channels 1, 2, 3)
                rgb_image = np.array(src.read([1, 2, 3]))
                transform = src.transform
                source_crs = src.crs.to_string() if src.crs else None

            # Transpose from (C, H, W) to (H, W, C)
            np_image = np.transpose(rgb_image, (1, 2, 0))
            pil_image = Image.fromarray(np_image.astype(np.uint8))
            
            # Import get_crs here to avoid circular import
            from ..geospatial.converter import get_crs
            source_crs = get_crs(image_path)  # Use the utility function

        elif ext in [".jpg", ".jpeg", ".png"]:
            # Load regular image
            pil_image = Image.open(image_path).convert("RGB")
            np_image = np.array(pil_image)
            source_crs = None  # No CRS for non-georeferenced images

        else:
            raise ValueError(f"Unsupported image format: {ext}")

        return ImageData(
            image_path=image_path,
            pil_image=pil_image,
            np_image=np_image,
            source_crs=source_crs,
            transform=transform,
            is_georeferenced=source_crs is not None,
        )

    @staticmethod
    def _load_from_array(image: np.ndarray) -> ImageData:
        """Load image from numpy array."""
        if image.ndim != 3 or image.shape[-1] not in [3, 4]:
            raise ValueError(
                f"Expected RGB(A) image with shape (H, W, 3) or (H, W, 4), "
                f"got shape {image.shape}"
            )

        # Drop alpha channel if present
        np_image = image[..., :3]

        # Ensure uint8
        if np_image.dtype != np.uint8:
            if np_image.max() <= 1.0:
                np_image = (np_image * 255).astype(np.uint8)
            else:
                np_image = np_image.astype(np.uint8)

        pil_image = Image.fromarray(np_image)

        return ImageData(
            image_path=None,
            pil_image=pil_image,
            np_image=np_image,
            source_crs=None,
            transform=None,
            is_georeferenced=False,
        )

    @staticmethod
    def _load_from_pil(image: Image.Image) -> ImageData:
        """Load image from PIL Image."""
        pil_image = image.convert("RGB")
        np_image = np.array(pil_image)

        return ImageData(
            image_path=None,
            pil_image=pil_image,
            np_image=np_image,
            source_crs=None,
            transform=None,
            is_georeferenced=False,
        )
