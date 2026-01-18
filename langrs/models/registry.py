"""Model registry for registering and retrieving model implementations."""

from typing import Dict, Type, Optional, List
from .base import DetectionModel, SegmentationModel


class ModelRegistry:
    """
    Registry for model implementations.
    
    This registry allows models to be registered and retrieved by name,
    enabling dynamic model selection and easy extension of the package.
    """

    _detection_models: Dict[str, Type[DetectionModel]] = {}
    _segmentation_models: Dict[str, Type[SegmentationModel]] = {}

    @classmethod
    def register_detection(cls, name: str):
        """
        Decorator to register a detection model class.
        
        Args:
            name: Unique name identifier for the model
            
        Returns:
            Decorator function
            
        Example:
            @ModelRegistry.register_detection("grounding_dino")
            class GroundingDINODetector(DetectionModel):
                ...
        """
        def decorator(model_class: Type[DetectionModel]) -> Type[DetectionModel]:
            if not issubclass(model_class, DetectionModel):
                raise TypeError(
                    f"Model class {model_class.__name__} must inherit from DetectionModel"
                )
            if name in cls._detection_models:
                raise ValueError(
                    f"Detection model '{name}' is already registered. "
                    f"Existing: {cls._detection_models[name].__name__}"
                )
            cls._detection_models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def register_segmentation(cls, name: str):
        """
        Decorator to register a segmentation model class.
        
        Args:
            name: Unique name identifier for the model
            
        Returns:
            Decorator function
            
        Example:
            @ModelRegistry.register_segmentation("sam")
            class SAMSegmenter(SegmentationModel):
                ...
        """
        def decorator(model_class: Type[SegmentationModel]) -> Type[SegmentationModel]:
            if not issubclass(model_class, SegmentationModel):
                raise TypeError(
                    f"Model class {model_class.__name__} must inherit from SegmentationModel"
                )
            if name in cls._segmentation_models:
                raise ValueError(
                    f"Segmentation model '{name}' is already registered. "
                    f"Existing: {cls._segmentation_models[name].__name__}"
                )
            cls._segmentation_models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_detection_model(cls, name: str) -> Optional[Type[DetectionModel]]:
        """
        Get a registered detection model class by name.
        
        Args:
            name: Name of the registered model
            
        Returns:
            Model class if found, None otherwise
        """
        return cls._detection_models.get(name)

    @classmethod
    def get_segmentation_model(cls, name: str) -> Optional[Type[SegmentationModel]]:
        """
        Get a registered segmentation model class by name.
        
        Args:
            name: Name of the registered model
            
        Returns:
            Model class if found, None otherwise
        """
        return cls._segmentation_models.get(name)

    @classmethod
    def list_detection_models(cls) -> List[str]:
        """
        List all registered detection model names.
        
        Returns:
            List of registered detection model names
        """
        return list(cls._detection_models.keys())

    @classmethod
    def list_segmentation_models(cls) -> List[str]:
        """
        List all registered segmentation model names.
        
        Returns:
            List of registered segmentation model names
        """
        return list(cls._segmentation_models.keys())

    @classmethod
    def is_detection_model_registered(cls, name: str) -> bool:
        """
        Check if a detection model is registered.
        
        Args:
            name: Name of the model to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._detection_models

    @classmethod
    def is_segmentation_model_registered(cls, name: str) -> bool:
        """
        Check if a segmentation model is registered.
        
        Args:
            name: Name of the model to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._segmentation_models

    @classmethod
    def unregister_detection(cls, name: str) -> bool:
        """
        Unregister a detection model.
        
        Args:
            name: Name of the model to unregister
            
        Returns:
            True if model was registered and removed, False otherwise
        """
        if name in cls._detection_models:
            del cls._detection_models[name]
            return True
        return False

    @classmethod
    def unregister_segmentation(cls, name: str) -> bool:
        """
        Unregister a segmentation model.
        
        Args:
            name: Name of the model to unregister
            
        Returns:
            True if model was registered and removed, False otherwise
        """
        if name in cls._segmentation_models:
            del cls._segmentation_models[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (useful for testing)."""
        cls._detection_models.clear()
        cls._segmentation_models.clear()
