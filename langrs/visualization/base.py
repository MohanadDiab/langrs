"""Abstract base class for visualization."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from ..utils.types import BoundingBox


class Visualizer(ABC):
    """
    Abstract base class for visualization.
    
    Different visualization backends can be implemented
    (matplotlib, PIL, etc.).
    """

    @abstractmethod
    def plot_boxes(
        self,
        image: Image.Image,
        boxes: List[BoundingBox],
        output_path: str,
        box_color: str = "r",
        linewidth: int = 1,
    ) -> None:
        """
        Plot bounding boxes on image.
        
        Args:
            image: Input image as PIL Image
            boxes: List of bounding boxes as (x_min, y_min, x_max, y_max)
            output_path: Path to save the output image
            box_color: Color of bounding boxes
            linewidth: Thickness of bounding box lines
        """
        pass

    @abstractmethod
    def plot_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        output_path: str,
        alpha: float = 0.4,
        colormap: str = "viridis",
    ) -> None:
        """
        Plot segmentation masks on image.
        
        Args:
            image: Input image as numpy array
            masks: Segmentation mask as numpy array
            output_path: Path to save the output image
            alpha: Transparency of mask overlay
            colormap: Colormap for mask visualization
        """
        pass

    @abstractmethod
    def plot_scatter(
        self,
        data: np.ndarray,
        outliers: Optional[np.ndarray] = None,
        title: str = "",
        output_path: str = "scatter.png",
        xlabel: str = "Index",
        ylabel: str = "Value",
    ) -> None:
        """
        Plot scatter plot with optional outlier highlighting.
        
        Args:
            data: Data points to plot
            outliers: Optional array of outlier indices
            title: Plot title
            output_path: Path to save the plot
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        pass
