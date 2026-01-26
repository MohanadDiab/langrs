"""
LangRS - Remote Sensing Image Segmentation Package

A modern, extensible package for remote sensing image segmentation using
foundation models like GroundingDINO and SAM.
"""

from typing import Optional

# Phase 1: Foundation - Base classes and interfaces
from .models.base import DetectionModel, SegmentationModel
from .models.registry import ModelRegistry

# Phase 2: Model implementations
try:
    from .models.detection.grounding_dino import GroundingDINODetector
except ImportError:
    GroundingDINODetector = None

try:
    from .models.detection.rex_omni import RexOmniDetector
except ImportError:
    RexOmniDetector = None

try:
    from .models.segmentation.sam import SAMSegmenter
except ImportError:
    SAMSegmenter = None

# Phase 3: Processing, visualization, I/O
from .processing.image_loader import ImageLoader, ImageData
from .processing.tiling import TilingStrategy, SlidingWindowTiler, Tile
from .processing.outlier_detection import (
    OutlierDetector,
    ZScoreOutlierDetector,
    IQROutlierDetector,
    RobustCovarianceOutlierDetector,
    SVMOutlierDetector,
    IsolationForestOutlierDetector,
    LOFOutlierDetector,
)
from .visualization.base import Visualizer
from .visualization.matplotlib_viz import MatplotlibVisualizer
from .io.output_manager import OutputManager

# Phase 4: Core pipeline
from .core.config import LangRSConfig
from .core.pipeline import LangRS
from .core.builder import LangRSBuilder
from .models.factory import ModelFactory

# Utilities
from .utils.exceptions import (
    LangRSError,
    ModelLoadError,
    ImageLoadError,
    DetectionError,
    SegmentationError,
    ConfigurationError,
)
from .processing.postprocessing import apply_nms, apply_nms_areas


# Public API
__all__ = [
    # Core pipeline
    "LangRS",
    "LangRSPipelineBuilder",
    "LangRSConfig",
    # Models
    "DetectionModel",
    "SegmentationModel",
    "ModelRegistry",
    "ModelFactory",
    "GroundingDINODetector",
    "RexOmniDetector",
    "SAMSegmenter",
    # Processing
    "ImageLoader",
    "ImageData",
    "TilingStrategy",
    "SlidingWindowTiler",
    "Tile",
    "OutlierDetector",
    "ZScoreOutlierDetector",
    "IQROutlierDetector",
    "RobustCovarianceOutlierDetector",
    "SVMOutlierDetector",
    "IsolationForestOutlierDetector",
    "LOFOutlierDetector",
    # Visualization
    "Visualizer",
    "MatplotlibVisualizer",
    # I/O
    "OutputManager",
    # Utilities
    "LangRSError",
    "ModelLoadError",
    "ImageLoadError",
    "DetectionError",
    "SegmentationError",
    "ConfigurationError",
    "apply_nms",
    "apply_nms_areas",
]
