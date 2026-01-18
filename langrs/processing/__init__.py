"""Processing modules for LangRS."""

from .image_loader import ImageLoader, ImageData
from .tiling import TilingStrategy, SlidingWindowTiler, Tile
from .outlier_detection import (
    OutlierDetector,
    ZScoreOutlierDetector,
    IQROutlierDetector,
    RobustCovarianceOutlierDetector,
    SVMOutlierDetector,
    IsolationForestOutlierDetector,
    LOFOutlierDetector,
)

__all__ = [
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
]
