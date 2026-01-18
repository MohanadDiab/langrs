"""Matplotlib-based visualization implementation."""

from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .base import Visualizer
from ..utils.types import BoundingBox


class MatplotlibVisualizer(Visualizer):
    """Matplotlib-based visualization implementation."""

    def __init__(self, figsize: tuple = (10, 10), dpi: int = 100):
        """
        Initialize matplotlib visualizer.
        
        Args:
            figsize: Default figure size (width, height)
            dpi: Resolution in dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_boxes(
        self,
        image: Image.Image,
        boxes: List[BoundingBox],
        output_path: str,
        box_color: str = "r",
        linewidth: int = 1,
    ) -> None:
        """Plot bounding boxes on image."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(image)
        ax.axis("off")

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=linewidth,
                edgecolor=box_color,
                facecolor="none",
            )
            ax.add_patch(rect)

        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=self.dpi)
        plt.close()

    def plot_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        output_path: str,
        alpha: float = 0.4,
        colormap: str = "viridis",
    ) -> None:
        """Plot segmentation masks on image."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(image)
        ax.imshow(masks, cmap=colormap, alpha=alpha)
        ax.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=self.dpi)
        plt.close()

    def plot_scatter(
        self,
        data: np.ndarray,
        outliers: Optional[np.ndarray] = None,
        title: str = "",
        output_path: str = "scatter.png",
        xlabel: str = "Index",
        ylabel: str = "Value",
    ) -> None:
        """Plot scatter plot with optional outlier highlighting."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        data_flat = data.flatten() if data.ndim > 1 else data
        indices = np.arange(len(data_flat))
        
        # Plot all points
        ax.scatter(indices, data_flat, c="blue", label="Data", alpha=0.6)
        
        # Highlight outliers if provided
        if outliers is not None:
            outliers_flat = outliers.flatten() if outliers.ndim > 1 else outliers
            outlier_indices = np.where(outliers_flat == -1)[0]
            if len(outlier_indices) > 0:
                ax.scatter(
                    outlier_indices,
                    data_flat[outlier_indices],
                    c="red",
                    label="Outliers",
                    alpha=0.8,
                )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if outliers is not None:
            ax.legend()
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
