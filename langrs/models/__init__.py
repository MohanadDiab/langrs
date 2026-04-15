"""Model abstractions and implementations for LangRS."""

from .base import DetectionModel, SegmentationModel
from .registry import ModelRegistry

# Register default models
try:
    from .detection.grounding_dino import GroundingDINODetector
    ModelRegistry.register_detection("grounding_dino")(GroundingDINODetector)
except ImportError:
    pass

try:
    from .detection.rex_omni import RexOmniDetector
    ModelRegistry.register_detection("rex_omni")(RexOmniDetector)
except ImportError:
    pass

try:
    from .segmentation.sam import SAMSegmenter
    ModelRegistry.register_segmentation("sam")(SAMSegmenter)
except ImportError:
    pass

__all__ = [
    "DetectionModel",
    "SegmentationModel",
    "ModelRegistry",
]
