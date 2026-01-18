"""Abstract base classes for detection and segmentation models."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import torch

from ..utils.types import BoundingBox, Device


class DetectionModel(ABC):
    """
    Abstract base class for object detection models.
    
    All detection models must implement this interface to be used
    with the LangRS pipeline.
    """

    @abstractmethod
    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ) -> List[BoundingBox]:
        """
        Detect objects in an image based on text prompt.
        
        Args:
            image: Input image as PIL Image or numpy array
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for bounding boxes (0.0-1.0)
            text_threshold: Confidence threshold for text matching (0.0-1.0)
            
        Returns:
            List of bounding boxes as (x_min, y_min, x_max, y_max) tuples
            
        Raises:
            DetectionError: If detection fails
        """
        pass

    @abstractmethod
    def load_weights(self, model_path: Optional[str] = None) -> None:
        """
        Load model weights from path or Hugging Face.
        
        Args:
            model_path: Optional path to model weights. If None, downloads
                       from Hugging Face or uses default location.
                       
        Raises:
            ModelLoadError: If model loading fails
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get the device the model is running on.
        
        Returns:
            torch.device: The device (CPU or CUDA)
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if model weights are loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        pass


class SegmentationModel(ABC):
    """
    Abstract base class for segmentation models.
    
    All segmentation models must implement this interface to be used
    with the LangRS pipeline.
    """

    @abstractmethod
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate segmentation masks for given bounding boxes.
        
        Args:
            image: Input image as PIL Image or numpy array
            boxes: Bounding boxes tensor of shape (N, 4) where each box
                  is (x_min, y_min, x_max, y_max)
                  
        Returns:
            Masks tensor of shape (N, H, W) where each mask is binary
            (0 or 1) indicating object pixels
            
        Raises:
            SegmentationError: If segmentation fails
        """
        pass

    @abstractmethod
    def load_weights(self, model_path: Optional[str] = None) -> None:
        """
        Load model weights from path or Hugging Face.
        
        Args:
            model_path: Optional path to model weights. If None, downloads
                       from Hugging Face or uses default location.
                       
        Raises:
            ModelLoadError: If model loading fails
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get the device the model is running on.
        
        Returns:
            torch.device: The device (CPU or CUDA)
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if model weights are loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        pass

    def set_image(self, image: Union[Image.Image, np.ndarray]) -> None:
        """
        Set the image for segmentation (for models that require image preprocessing).
        
        This is an optional method that some models may need to preprocess
        the image before segmentation. Default implementation does nothing.
        
        Args:
            image: Input image as PIL Image or numpy array
        """
        pass
