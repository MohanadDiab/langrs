"""Factory for creating model instances."""

from typing import Optional, Union
import torch

from .base import DetectionModel, SegmentationModel
from .registry import ModelRegistry
from ..utils.exceptions import ModelLoadError


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_detection_model(
        model_name: str = "grounding_dino",
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> DetectionModel:
        """
        Create a detection model instance.
        
        Args:
            model_name: Name of the registered detection model
            model_path: Optional path to model checkpoint
            device: Optional device ('cpu', 'cuda', or torch.device)
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            DetectionModel instance
            
        Raises:
            ModelLoadError: If model class not found or creation fails
        """
        model_class = ModelRegistry.get_detection_model(model_name)
        if model_class is None:
            available = ModelRegistry.list_detection_models()
            raise ModelLoadError(
                f"Unknown detection model: {model_name}. "
                f"Available models: {available}"
            )

        try:
            if device is not None:
                kwargs["device"] = device
            if model_path is not None:
                kwargs["model_path"] = model_path

            return model_class(**kwargs)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to create detection model '{model_name}': {e}"
            ) from e

    @staticmethod
    def create_segmentation_model(
        model_name: str = "sam",
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> SegmentationModel:
        """
        Create a segmentation model instance.
        
        Args:
            model_name: Name of the registered segmentation model
            model_path: Optional path to model checkpoint
            device: Optional device ('cpu', 'cuda', or torch.device)
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            SegmentationModel instance
            
        Raises:
            ModelLoadError: If model class not found or creation fails
        """
        model_class = ModelRegistry.get_segmentation_model(model_name)
        if model_class is None:
            available = ModelRegistry.list_segmentation_models()
            raise ModelLoadError(
                f"Unknown segmentation model: {model_name}. "
                f"Available models: {available}"
            )

        try:
            if device is not None:
                kwargs["device"] = device
            if model_path is not None:
                kwargs["model_path"] = model_path

            return model_class(**kwargs)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to create segmentation model '{model_name}': {e}"
            ) from e
