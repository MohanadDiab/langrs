"""Tests for visualization module."""

import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from langrs.visualization.base import Visualizer
from langrs.visualization.matplotlib_viz import MatplotlibVisualizer


class TestVisualizer:
    """Test Visualizer ABC."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            Visualizer()


class TestMatplotlibVisualizer:
    """Test MatplotlibVisualizer implementation."""

    def test_initialization(self):
        """Test visualizer initialization."""
        viz = MatplotlibVisualizer()
        assert viz.figsize == (10, 10)
        assert viz.dpi == 100

    def test_initialization_custom(self):
        """Test custom initialization."""
        viz = MatplotlibVisualizer(figsize=(15, 15), dpi=150)
        assert viz.figsize == (15, 15)
        assert viz.dpi == 150

    def test_plot_boxes(self, sample_image):
        """Test plotting bounding boxes."""
        viz = MatplotlibVisualizer()
        boxes = [(10, 10, 50, 50), (60, 60, 90, 90)]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            viz.plot_boxes(sample_image, boxes, output_path)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_plot_boxes_custom_style(self, sample_image):
        """Test plotting boxes with custom style."""
        viz = MatplotlibVisualizer()
        boxes = [(10, 10, 50, 50)]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            viz.plot_boxes(
                sample_image, boxes, output_path, box_color="blue", linewidth=2
            )
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_plot_masks(self, sample_numpy_image):
        """Test plotting masks."""
        viz = MatplotlibVisualizer()
        masks = np.zeros((100, 100), dtype=np.uint8)
        masks[40:60, 40:60] = 255  # Small square mask

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            viz.plot_masks(sample_numpy_image, masks, output_path)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_plot_masks_custom_style(self, sample_numpy_image):
        """Test plotting masks with custom style."""
        viz = MatplotlibVisualizer()
        masks = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            viz.plot_masks(
                sample_numpy_image, masks, output_path, alpha=0.6, colormap="plasma"
            )
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_plot_scatter(self):
        """Test plotting scatter plot."""
        viz = MatplotlibVisualizer()
        data = np.array([1, 2, 3, 4, 5, 100])  # Last is outlier

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            viz.plot_scatter(data, title="Test Scatter", output_path=output_path)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_plot_scatter_with_outliers(self):
        """Test plotting scatter with outlier highlighting."""
        viz = MatplotlibVisualizer()
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = np.array([1, 1, 1, 1, 1, -1])  # Last is outlier

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            viz.plot_scatter(
                data, outliers=outliers, title="Test with Outliers", output_path=output_path
            )
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_plot_scatter_2d_array(self):
        """Test plotting scatter with 2D array."""
        viz = MatplotlibVisualizer()
        data = np.array([[1], [2], [3], [4], [5]])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            viz.plot_scatter(data, output_path=output_path)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()
