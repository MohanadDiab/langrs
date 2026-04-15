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
    from .models.segmentation.sam import SAMSegmenter
except ImportError:
    SAMSegmenter = None

# Phase 3: Processing, visualization, I/O
try:
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
except ImportError:
    # Optional runtime deps (e.g. rasterio/geospatial stack) may not be present in
    # minimal environments. Import submodules directly as needed.
    ImageLoader = None
    ImageData = None
    TilingStrategy = None
    SlidingWindowTiler = None
    Tile = None
    OutlierDetector = None
    ZScoreOutlierDetector = None
    IQROutlierDetector = None
    RobustCovarianceOutlierDetector = None
    SVMOutlierDetector = None
    IsolationForestOutlierDetector = None
    LOFOutlierDetector = None
    Visualizer = None
    MatplotlibVisualizer = None
    OutputManager = None

# Phase 4: Core pipeline
try:
    from .core.config import LangRSConfig
    from .core.pipeline import LangRS
    from .core.builder import LangRSBuilder
    from .models.factory import ModelFactory
except ImportError:
    LangRSConfig = None
    LangRS = None
    LangRSBuilder = None
    ModelFactory = None

# Backwards-compatible alias (older docs referenced LangRSPipelineBuilder).
LangRSPipelineBuilder = LangRSBuilder

# Utilities
from .utils.exceptions import (
    LangRSError,
    ModelLoadError,
    ImageLoadError,
    DetectionError,
    SegmentationError,
    ConfigurationError,
)
try:
    from .processing.postprocessing import apply_nms, apply_nms_areas
except ImportError:
    apply_nms = None
    apply_nms_areas = None


# Public API
__all__ = [
    # Core pipeline
    "LangRS",
    "LangRSBuilder",
    "LangRSPipelineBuilder",
    "LangRSConfig",
    # Models
    "DetectionModel",
    "SegmentationModel",
    "ModelRegistry",
    "ModelFactory",
    "GroundingDINODetector",
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
